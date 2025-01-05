// Output of the vertex shader for pixel shader
struct VSOUTPUT
{
	float4 position : SV_Position;
	float4 color : COLOR0;
};

// Constant buffer for projection matrix
struct Projection
{
	float4x4 data;
};
// using Binding 0, Set 0
ConstantBuffer<Projection> projection : register(b0, space0);

// Constant buffer for transform matrix
struct Transforms
{
	float4x4 data[3];
};
// using Binding 1, Set 0
ConstantBuffer<Transforms> transforms : register(b0, space1);

VSOUTPUT main(uint vertex_id : SV_VertexID)
{
	// Triangle vertices
	float3 tri_pos[3] = {
		{0.5f, -0.5f, 1.5f},
		{-0.5f, -0.5f, 1.5f},
		{0.f, 0.5f, 1.5f},
	};

	// Triangle colors per vertex
	float4 tri_col[3] = {
		{1.f, 0.f, 0.f, 1.f},
		{0.f, 1.f, 0.f, 1.f},
		{0.f, 0.f, 1.f, 1.f},
	};

	// select transform matrix using instance_id
	float4x4 transform = transforms.data[0];

	// select vertex position using vertex_id
	float4 pos = float4(tri_pos[vertex_id], 1.f);

	// transform vertex position by transform matrix
	pos = mul(transform, pos);

	// transform vertex position by projection matrix
	pos = mul(projection.data, pos);

	VSOUTPUT output = {
		pos,
		tri_col[vertex_id]
	};

	return output;
}