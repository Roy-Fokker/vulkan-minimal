// Pixel shader input structure, mirroring VSOUTPUT
struct PSINPUT
{
	float4 position : SV_Position;
	float4 color : COLOR;
	float2 uv : TEXCOORD0;
};

Texture2D<float4> textureObj : register(t0, space2);
SamplerState textureSampler : register(s0, space2);

float4 main(PSINPUT input) : SV_Target
{
	return input.color;
}