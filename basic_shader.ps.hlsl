struct PSINPUT
{
	float4 position : SV_Position;
	float4 color : COLOR;
};

float4 main(PSINPUT input) : SV_Target
{
	return input.color;
}