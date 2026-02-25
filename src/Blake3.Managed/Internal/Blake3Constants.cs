namespace Blake3.Managed.Internal;

internal static class Blake3Constants
{
    public const int BlockLen = 64;
    public const int ChunkLen = 1024;
    public const int KeyLen = 32;
    public const int OutLen = 32;
    public const int MaxDepth = 32; // supports 2^32 chunks = 4 TB; was 54 (18 PB)

    // Domain separation flags
    public const uint ChunkStart = 1 << 0;
    public const uint ChunkEnd = 1 << 1;
    public const uint Parent = 1 << 2;
    public const uint Root = 1 << 3;
    public const uint KeyedHash = 1 << 4;
    public const uint DeriveKeyContext = 1 << 5;
    public const uint DeriveKeyMaterial = 1 << 6;

    // IV words (same as SHA-256)
    public static ReadOnlySpan<uint> IV => new uint[]
    {
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
    };

    // Message word permutation for each round
    public static ReadOnlySpan<byte> MsgPermutation => new byte[]
    {
        2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8
    };

    // Precomputed 7-round message schedule (avoids runtime permutation)
    // Round 0: identity [0..15]
    // Round 1..6: apply MsgPermutation iteratively
    public static ReadOnlySpan<byte> MsgSchedule => new byte[]
    {
        // Round 0
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        // Round 1
        2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8,
        // Round 2
        3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1,
        // Round 3
        10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6,
        // Round 4
        12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4,
        // Round 5
        9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7,
        // Round 6
        11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13,
    };
}
