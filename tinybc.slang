// #define SIMPLEX_VIS

uniform int2 textureDim;

// Optimized Nelder-Mead parameters (balanced for 64 iterations)
static const float ALPHA = 1.3;  // reflection coefficient (moderately aggressive)
static const float GAMMA = 2.2;  // expansion coefficient (balanced exploration)
static const float RHO = 0.55;   // contraction coefficient (slightly more aggressive)
static const float SIGMA = 0.45; // shrink coefficient (balanced shrinking)
static const int MAX_ITERATIONS = 64;
static const float GOOD_ENOUGH_LOSS = 0.004; // threshold for "good enough" initial solution

// Uncompressed input texture
Texture2D<float4> gInputTex;

// Decoded BC texture
RWTexture2D<float4> gDecodedTex;

struct SimplexVertex
{
    float4 minEndPoints;
    float4 maxEndPoints;
    float score; // loss value
};

struct line4
{
    float4 a;
    float4 b;
};

float4 normalize_safe(float4 a, float4 safe)
{
    float length = dot(a, a);
    if (length != 0.0)
    {
        return a / sqrt(length);
    }
    else
    {
        return safe;
    }
}

float4 get_texel_color_from_block(float4 block_texels[16], uint texel_index)
{
    float4 texel_color = block_texels[texel_index];
    texel_color.rgb = texel_color.rgb * texel_color.w;
    return texel_color;
}

// check compute_ideal_colors_and_weights_4_comp
void compute_unorm_end_point_and_unorm_weight(float4 block_texels[16], out float4 endpt0, out float4 endpt1, out float weights[16])
{
    float4 average = 0;
    {
        for (uint texel_index = 0; texel_index < 16; texel_index++)
        {
            float4 texel_color = get_texel_color_from_block(block_texels, texel_index);
            average += texel_color;
        }
        average /= 16;
    }

    float4 dir;
    {
        float4 sum_xp = 0;
        float4 sum_yp = 0;
        float4 sum_zp = 0;
        float4 sum_wp = 0;
        for (uint texel_index = 0; texel_index < 16; texel_index++)
        {
            float4 texel_color = get_texel_color_from_block(block_texels, texel_index);
            float4 texel_datum = texel_color - average;
            sum_xp += texel_datum.x > 0 ? texel_datum : 0;
            sum_yp += texel_datum.y > 0 ? texel_datum : 0;
            sum_zp += texel_datum.z > 0 ? texel_datum : 0;
            sum_wp += texel_datum.w > 0 ? texel_datum : 0;
        }

        float prod_xp = dot(sum_xp, sum_xp);
        float prod_yp = dot(sum_yp, sum_yp);
        float prod_zp = dot(sum_zp, sum_zp);
        float prod_wp = dot(sum_wp, sum_wp);

        float4 best_vector = sum_xp;
        float best_sum = prod_xp;

        if (prod_yp > best_sum)
        {
            best_vector = sum_yp;
            best_sum = prod_yp;
        }

        if (prod_zp > best_sum)
        {
            best_vector = sum_zp;
            best_sum = prod_zp;
        }

        dir = best_vector;

        if ((dir.x + dir.y + dir.z + dir.w) < 0.0f)
        {
            dir = -dir;
        }
    }

    {
        line4 l;
        l.a = average;
        l.b = normalize_safe(dir, float4(0, 0, 0, 0));
        float lowparam = 1e10f;
        float highparam = -1e10f;
        {
            for (uint texel_index = 0; texel_index < 16; texel_index++)
            {
                float4 texel = get_texel_color_from_block(block_texels, texel_index);
                float param = dot(texel - l.a, l.b);
                weights[texel_index] = param;

                lowparam = min(param, lowparam);
                highparam = max(param, highparam);
            }
        }

        if (highparam <= lowparam)
        {
            lowparam = 0.0f;
            highparam = 1e-7f;
        }

        float length = highparam - lowparam;
        float scale = 1.0f / length;

        float4 offset_a = l.b * lowparam;
        float4 offset_b = l.b * highparam;

        endpt0 = saturate(l.a + min(offset_a, offset_b));
        endpt1 = saturate(l.a + max(offset_a, offset_b));

        {
            for (uint texel_index = 0; texel_index < 16; texel_index++)
            {
                weights[texel_index] = saturate((weights[texel_index] - lowparam) * scale);
            }
        }
    }
}

// Compute weights deterministically given endpoints
void compute_weights(float4 minEndPoints, float4 maxEndPoints, float4 target[16], out float weights[16])
{
    float4 dir = maxEndPoints - minEndPoints;
    float dirLength = length(dir);
    
    if (dirLength < 1e-6)
    {
        // Degenerate case - all weights become 0.5
        for (int i = 0; i < 16; i++)
            weights[i] = 0.5;
        return;
    }
    
    dir /= dirLength;
    
    for (int i = 0; i < 16; i++)
    {
        float proj = dot(target[i] - minEndPoints, dir);
        weights[i] = saturate(proj / dirLength);
    }
}

// Compute total loss for a given set of endpoints
float compute_loss(float4 minEndPoints, float4 maxEndPoints, float4 target[16])
{
    float totalLoss = 0;
    float weights[16];
    compute_weights(minEndPoints, maxEndPoints, target, weights);
    
    for (int i = 0; i < 16; i++)
    {
        float4 decoded = lerp(minEndPoints, maxEndPoints, weights[i]);
        float4 delta = target[i] - decoded;
        totalLoss += dot(delta, delta);
    }
    return totalLoss;
}

// Sort simplex vertices by score (ascending)
void sort_simplex(inout SimplexVertex simplex[9])
{
    // Simple bubble sort
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8 - i; j++)
        {
            if (simplex[j].score > simplex[j + 1].score)
            {
                SimplexVertex temp = simplex[j];
                simplex[j] = simplex[j + 1];
                simplex[j + 1] = temp;
            }
        }
    }
}

// Compute centroid of all points except the worst
float4 compute_centroid(SimplexVertex simplex[9], bool isMin)
{
    float4 centroid = 0;
    for (int i = 0; i < 8; i++)
    {
        centroid += isMin ? simplex[i].minEndPoints : simplex[i].maxEndPoints;
    }
    return centroid / 8.0;
}

void encoder(uint2 tileId)
{
    let texel = 4 * tileId;
    if (any(texel >= textureDim)) return;

    // Load target values
    float4 target[16];
    float4 minEndPoints = float4(1);
    float4 maxEndPoints = float4(0);
    
    for (int i = 0; i < 16; i++)
    {
        target[i] = gInputTex[texel + int2(i % 4, i / 4)];
        minEndPoints = min(minEndPoints, target[i]);
        maxEndPoints = max(maxEndPoints, target[i]);
    }

    // Initialize better endpoints using PCA-like method
    float4 endPt0, endPt1;
    float initialWeights[16];
    compute_unorm_end_point_and_unorm_weight(target, endPt0, endPt1, initialWeights);
    minEndPoints.xyz = endPt0.xyz;
    maxEndPoints.xyz = endPt1.xyz;

    // Check if PCA result is good enough
    float initialLoss = compute_loss(minEndPoints, maxEndPoints, target);
    bool needsSimplex = initialLoss > GOOD_ENOUGH_LOSS;
    
    if (!needsSimplex)
    {
        // PCA result is good enough, skip Nelder-Mead
        float finalWeights[16];
        compute_weights(minEndPoints, maxEndPoints, target, finalWeights);
        
        for (int i = 0; i < 16; i++)
        {
            float4 decoded = lerp(minEndPoints, maxEndPoints, finalWeights[i]);
            gDecodedTex[texel + int2(i % 4, i / 4)] = decoded;
        }
        return;
    }

#if 1
    // Initialize simplex with 9 vertices (8D + 1)
    SimplexVertex simplex[9];
    simplex[0].minEndPoints = minEndPoints;
    simplex[0].maxEndPoints = maxEndPoints;
    simplex[0].score = initialLoss;  // reuse computed loss

    // Generate other vertices with small perturbations
    for (int i = 1; i < 9; i++)
    {
        simplex[i].minEndPoints = minEndPoints;
        simplex[i].maxEndPoints = maxEndPoints;
        
        // Perturb either min or max endpoints
        if (i <= 4)
            simplex[i].minEndPoints[i-1] = clamp(minEndPoints[i-1] + 0.1, 0, 1);
        else
            simplex[i].maxEndPoints[i-5] = clamp(maxEndPoints[i-5] + 0.1, 0, 1);
            
        simplex[i].score = compute_loss(simplex[i].minEndPoints, simplex[i].maxEndPoints, target);
    }

    // Main Nelder-Mead optimization loop
    for (int iter = 0; iter < MAX_ITERATIONS; iter++)
    {
        sort_simplex(simplex);
        
        // Compute centroid of all points except worst
        float4 centroidMin = compute_centroid(simplex, true);
        float4 centroidMax = compute_centroid(simplex, false);
        
        // Reflection
        float4 reflectedMin = centroidMin + ALPHA * (centroidMin - simplex[8].minEndPoints);
        float4 reflectedMax = centroidMax + ALPHA * (centroidMax - simplex[8].maxEndPoints);
        reflectedMin = clamp(reflectedMin, 0, 1);
        reflectedMax = clamp(reflectedMax, 0, 1);
        
        float reflectedScore = compute_loss(reflectedMin, reflectedMax, target);
        
        if (reflectedScore < simplex[0].score)
        {
            // Expansion
            float4 expandedMin = centroidMin + GAMMA * (reflectedMin - centroidMin);
            float4 expandedMax = centroidMax + GAMMA * (reflectedMax - centroidMax);
            expandedMin = clamp(expandedMin, 0, 1);
            expandedMax = clamp(expandedMax, 0, 1);
            
            float expandedScore = compute_loss(expandedMin, expandedMax, target);
            
            if (expandedScore < reflectedScore)
            {
                simplex[8].minEndPoints = expandedMin;
                simplex[8].maxEndPoints = expandedMax;
                simplex[8].score = expandedScore;
            }
            else
            {
                simplex[8].minEndPoints = reflectedMin;
                simplex[8].maxEndPoints = reflectedMax;
                simplex[8].score = reflectedScore;
            }
        }
        else if (reflectedScore < simplex[7].score)
        {
            simplex[8].minEndPoints = reflectedMin;
            simplex[8].maxEndPoints = reflectedMax;
            simplex[8].score = reflectedScore;
        }
        else
        {
            // Contraction
            float4 contractedMin = centroidMin + RHO * (simplex[8].minEndPoints - centroidMin);
            float4 contractedMax = centroidMax + RHO * (simplex[8].maxEndPoints - centroidMax);
            contractedMin = clamp(contractedMin, 0, 1);
            contractedMax = clamp(contractedMax, 0, 1);
            
            float contractedScore = compute_loss(contractedMin, contractedMax, target);
            
            if (contractedScore < simplex[8].score)
            {
                simplex[8].minEndPoints = contractedMin;
                simplex[8].maxEndPoints = contractedMax;
                simplex[8].score = contractedScore;
            }
            else
            {
                // Shrink
                for (int i = 1; i < 9; i++)
                {
                    simplex[i].minEndPoints = simplex[0].minEndPoints + SIGMA * (simplex[i].minEndPoints - simplex[0].minEndPoints);
                    simplex[i].maxEndPoints = simplex[0].maxEndPoints + SIGMA * (simplex[i].maxEndPoints - simplex[0].maxEndPoints);
                    simplex[i].minEndPoints = clamp(simplex[i].minEndPoints, 0, 1);
                    simplex[i].maxEndPoints = clamp(simplex[i].maxEndPoints, 0, 1);
                    simplex[i].score = compute_loss(simplex[i].minEndPoints, simplex[i].maxEndPoints, target);
                }
            }
        }
    }

    // Use best solution found
    sort_simplex(simplex);
    minEndPoints = simplex[0].minEndPoints;
    maxEndPoints = simplex[0].maxEndPoints;
#endif

    // Compute final weights and write output
    float finalWeights[16];
    compute_weights(minEndPoints, maxEndPoints, target, finalWeights);
    
    for (int i = 0; i < 16; i++)
    {
        float4 decoded = lerp(minEndPoints, maxEndPoints, finalWeights[i]);
#ifdef SIMPLEX_VIS
        decoded.rgb = float3(1, 0, 0);  // Mark Simplex blocks in red
#endif
        gDecodedTex[texel + int2(i % 4, i / 4)] = decoded;
    }
}
