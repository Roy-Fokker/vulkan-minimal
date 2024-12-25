struct VSOUTPUT
{
	float4 position : SV_Position;
	float4 color : COLOR0;
};

struct Projection
{
	float4x4 data;
};

ConstantBuffer<Projection> projection : register(b0, space0);

VSOUTPUT main(uint vertex_id : SV_VertexID)
{
	float3 tri_pos[3] = {
		{0.5f, -0.5f, 1.5f},
		{-0.5f, -0.5f, 1.5f},
		{0.f, 0.5f, 1.5f},
	};

	float4 tri_col[3] = {
		{1.f, 0.f, 0.f, 1.f},
		{0.f, 1.f, 0.f, 1.f},
		{0.f, 0.f, 1.f, 1.f},
	};

	float4 pos = float4(tri_pos[vertex_id], 1.f);
	pos = mul(projection.data, pos);

	VSOUTPUT output = {
		pos,
		tri_col[vertex_id]
	};
	

	return output;
}