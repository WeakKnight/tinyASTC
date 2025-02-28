static const bool kUseAdam = true;
static const int kNumOptimizationSteps = 16;

static const float lr = 0.0155;
static const float adamBeta1 = 0.78;
static const float adamBeta2 = 0.999;

// static const float lr = 0.1;
// static const float adamBeta1 = 0.9;
// static const float adamBeta2 = 0.999;

uniform int2 textureDim;


// Uncompressed input texture
Texture2D<float4> gInputTex;

// Decoded BC texture
RWTexture2D<float4> gDecodedTex;

struct line4
{
    float4 a;
    float4 b;
};

struct BCWeights : IDifferentiable
{
    float w[16];

    __init(float _w)
    {
        for (int i = 0; i < 16; i++)
            w[i] = _w;
    }

    __init(float _w[16])
    {
        for (int i = 0; i < 16; i++)
            w[i] = _w[i];
    }

    [mutating]
    void clamp()
    {
        for (int i = 0; i < 16; i++)
            w[i] = clamp(w[i], 0, 1);
    }

    [mutating]
    void quantize()
    {
        for (int i = 0; i < 15; i++)
            w[i] = round(w[i] * 64) / 64;
    }

    BCWeights rsqrt()
    {
        BCWeights r;
        for (int i = 0; i < 16; i++)
            r.w[i] = rsqrt(w[i]);
        return r;
    }
}

BCWeights operator +(float a, BCWeights b)
{
    BCWeights r;
    for (int i = 0; i < 16; i++)
        r.w[i] = a + b.w[i];
    return r;
}

BCWeights operator +(BCWeights a, BCWeights b)
{
    BCWeights r;
    for (int i = 0; i < 16; i++)
        r.w[i] = a.w[i] + b.w[i];
    return r;
}

BCWeights operator -(BCWeights a, BCWeights b)
{
    BCWeights r;
    for (int i = 0; i < 16; i++)
        r.w[i] = a.w[i] - b.w[i];
    return r;
}

BCWeights operator *(BCWeights a, BCWeights b)
{
    BCWeights r;
    for (int i = 0; i < 16; i++)
        r.w[i] = a.w[i] * b.w[i];
    return r;
}

BCWeights operator *(float a, BCWeights b)
{
    BCWeights r;
    for (int i = 0; i < 16; i++)
        r.w[i] = a * b.w[i];
    return r;
}

// This struct holds all of the data for a 4x4 texel tile
struct BCTile : IDifferentiable
{
    BCWeights weights;
    float4 minEndPoints;
    float4 maxEndPoints;

    __init()
    {
        weights = BCWeights(0.f);
        minEndPoints = 0.f;
        maxEndPoints = 0.f;
    }

    __init(BCTile.Differential tile)
    {
        weights = BCWeights(tile.weights.w);
        minEndPoints = tile.minEndPoints;
        maxEndPoints = tile.maxEndPoints;
    }

    __init(float _weight, float4 _minEndPoints, float4 _maxEndPoints)
    {
        weights = BCWeights(_weight);
        minEndPoints = _minEndPoints;
        maxEndPoints = _maxEndPoints;
    }

    [mutating]
    void clamp()
    {
        weights.clamp();
        minEndPoints = clamp(minEndPoints, 0, 1);
        maxEndPoints = clamp(maxEndPoints, 0, 1);
    }

    [mutating]
    void quantize()
    {
        weights.quantize();
        minEndPoints = round(minEndPoints * 127) / 127;
        maxEndPoints = round(maxEndPoints * 127) / 127;
    }

    BCTile rsqrt()
    {
        BCTile r;
        r.weights = weights.rsqrt();
        r.minEndPoints = rsqrt(minEndPoints);
        r.maxEndPoints = rsqrt(maxEndPoints);
        return r;
    }

    [BackwardDifferentiable]
    float4 decode(int i)
    {
        return lerp(minEndPoints, maxEndPoints, weights.w[i]);
    }
}

BCTile operator *(BCTile a, BCTile b)
{
    BCTile r;
    r.weights = a.weights * b.weights;
    r.minEndPoints = a.minEndPoints * b.minEndPoints;
    r.maxEndPoints = a.maxEndPoints * b.maxEndPoints;
    return r;
}

BCTile operator *(float a, BCTile b)
{
    BCTile r;
    r.weights = a * b.weights;
    r.minEndPoints = a * b.minEndPoints;
    r.maxEndPoints = a * b.maxEndPoints;
    return r;
}

BCTile operator +(float a, BCTile b)
{
    BCTile r;
    r.weights = a + b.weights;
    r.minEndPoints = a + b.minEndPoints;
    r.maxEndPoints = a + b.maxEndPoints;
    return r;
}

BCTile operator +(BCTile a, BCTile b)
{
    BCTile r;
    r.weights = a.weights + b.weights;
    r.minEndPoints = a.minEndPoints + b.minEndPoints;
    r.maxEndPoints = a.maxEndPoints + b.maxEndPoints;
    return r;
}

BCTile operator -(BCTile a, BCTile b)
{
    BCTile r;
    r.weights = a.weights - b.weights;
    r.minEndPoints = a.minEndPoints - b.minEndPoints;
    r.maxEndPoints = a.maxEndPoints - b.maxEndPoints;
    return r;
}

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

[BackwardDifferentiable]
void l2Loss(BCTile tile, no_diff float4 target[16], out float loss[16])
{
    [ForceUnroll]
    for (int i = 0; i < 16; i++)
    {
        let delta = target[i] - tile.decode(i);
        loss[i] = dot(delta, delta);
    }
}

void encoder(uint2 tileId)
{
    let texel = 4 * tileId;
    if (any(texel >= textureDim)) return;

    // Adam moments
    BCTile m1, m2;

    // Load target values and initialize with mid-point interpolation
    // and minmax endpoints computed over the tile
    float4 target[16];
    BCTile tile = BCTile(0.5, float4(1), float4(0));    
    for (int i = 0; i < 16; i++)
    {
        target[i] = gInputTex[texel + int2(i % 4, i / 4)];
        tile.minEndPoints = min(tile.minEndPoints, target[i]);
        tile.maxEndPoints = max(tile.maxEndPoints, target[i]);
    }
#if 1 // Better Initial Value
    float4 endPt0, endPt1;
    float weights[16];
    compute_unorm_end_point_and_unorm_weight(target, endPt0, endPt1, weights);
    tile.minEndPoints.xyz = endPt0.xyz;
    tile.maxEndPoints.xyz = endPt1.xyz;
    tile.weights = BCWeights(weights);
#endif

#if 1
    // To compress the texture we perform a number of gradient descent steps
    // to joint-optimize interpolation weights and endpoints for each tile.
    // Eventually the interpolation weights are quantized and frozen,
    // while the endpoints are fine tuned for a few more steps.
    bool frozenWeight = false;
    let endPointsFineTuninigThres = 0.85f;
    for (int s = 0; s < kNumOptimizationSteps; s++)
    {
        // Forward pass - compute error
        float loss[16];
        l2Loss(tile, target, loss);

        // Backward pass - compute gradients
        var dp_tile = diffPair(tile);
        __bwd_diff(l2Loss)(dp_tile, target, {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1});

        // Use Adam optimizer to update gradients
        BCTile grad = BCTile(dp_tile.d);
        if (kUseAdam)
        {
            m1 = adamBeta1 * m1 + (1 - adamBeta1) * grad;
            m2 = adamBeta2 * m2 + (1 - adamBeta2) * (grad * grad);
            let unbiased_m1 = rcp(1.f - pow(adamBeta1, 1 + s)) * m1;
            let unbiased_m2 = rcp(1.f - pow(adamBeta2, 1 + s)) * m2;
            grad = unbiased_m1 * (1E-10f + unbiased_m2).rsqrt();
        }

        // If weights are frozen zero their gradient
        if (frozenWeight)
            grad.weights = BCWeights(0);

        // Update BC tile via gradient descent & clamp its values to a valid range
        tile = tile - lr * grad;
        tile.clamp();

        // Time to freeze the BC weights?
        if (frozenWeight == false && float(s) / kNumOptimizationSteps >= endPointsFineTuninigThres)
        {
            // Quantize & freeze weights
            // We will continue to fine tune the endpoints for a few more steps
            frozenWeight = true;
            tile.weights.quantize();
        }
    }
#endif

    // Quantize & write out decoded tile
    tile.quantize();
    for (int i = 0; i < 16; i++)
        gDecodedTex[texel + int2(i % 4, i / 4)] = tile.decode(i);;
}
