struct VSInput
{
    float3 inPosition;
    float3 inColor;
    float2 inTexCoord;
}

struct UniformBuffer
{
    float deltaTime;
}
ConstantBuffer<UniformBuffer> ubo;

struct Particle
{
    float2 position;
    float2 velocity;
    float4 color;
};

struct ParticleSSBO
{
    Particle particles;
};

StructuredBuffer<ParticleSSBO> particlesIn;
RWStructuredBuffer<ParticleSSBO> particlesOut;

struct VSOutput
{
    float4 pos : SV_Position;
    float3 color;
    float2 fragTexCoord;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void compMain(uint3 threadId: SV_DispatchThreadID)
{
    uint index = threadId.x;

    particlesOut[index].particles.position = particlesIn[index].particles.position + particlesIn[index].particles.velocity.xy * ubo.deltaTime;
    particlesOut[index].particles.velocity = particlesIn[index].particles.velocity;

    // Flip movement at window border
    if ((particlesOut[index].particles.position.x <= -1.0) || (particlesOut[index].particles.position.x >= 1.0))
    {
        particlesOut[index].particles.velocity.x = -particlesOut[index].particles.velocity.x;
    }
    if ((particlesOut[index].particles.position.y <= -1.0) || (particlesOut[index].particles.position.y >= 1.0))
    {
        particlesOut[index].particles.velocity.y = -particlesOut[index].particles.velocity.y;
    }
}

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
    VSOutput output;
    output.pos = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.inPosition, 1.0))));
    output.color = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    return output;
}

Sampler2D texture;

[shader("fragment")]
float4 fragMain(VSOutput vertIn)
    : SV_Target
{
    return texture.Sample(vertIn.fragTexCoord);
}
