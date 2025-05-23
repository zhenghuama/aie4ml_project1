const int N = 4; //Batch dimensions (rows of A)
const int K = 768; //Length-wise split (Inner dimension)
const int M = 128; //Height of layer (columns of B)
const int T = 24; //Number of tiles the K dimension is split onto
