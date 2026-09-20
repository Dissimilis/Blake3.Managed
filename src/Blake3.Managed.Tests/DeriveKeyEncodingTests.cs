using System.Text;
using Blake3.Managed;

namespace Blake3.Managed.Tests;

/// <summary>
/// The derive-key context string is specified as UTF-8, and both string overloads encode it
/// with <see cref="Encoding.UTF8"/> before handing it to the byte overload.
///
/// Nothing pins that today. Every derive-key context in the suite -- including the official
/// vectors -- is pure ASCII, where UTF-8, ASCII, Latin-1 and the system default all produce
/// identical bytes. Swapping the encoding would therefore pass the entire suite while silently
/// producing a different key from every other BLAKE3 implementation for any caller whose
/// context contains a non-ASCII character. There is no exception and no failing vector; the
/// two sides just disagree.
/// </summary>
public class DeriveKeyEncodingTests
{
    // Deliberately exercises characters that differ between the plausible wrong encodings:
    // a Latin-1 representable accent, a character outside Latin-1, and one outside the BMP.
    private const string NonAsciiContext = "Blake3.Managed café — ключ 🔑 2026";

    private static byte[] Input() => HasherTests.MakeTestInput(4096);

    [Fact]
    public void HasherDeriveKeyContextIsEncodedAsUtf8()
    {
        byte[] data = Input();

        using var fromString = Hasher.NewDeriveKey(NonAsciiContext);
        fromString.Update(data);

        using var fromUtf8 = Hasher.NewDeriveKey(Encoding.UTF8.GetBytes(NonAsciiContext));
        fromUtf8.Update(data);

        Assert.Equal(fromUtf8.Finalize().ToString(), fromString.Finalize().ToString());
    }

    [Fact]
    public void HasherDeriveKeyIsNotEncodedAsLatin1()
    {
        byte[] data = Input();

        using var fromString = Hasher.NewDeriveKey(NonAsciiContext);
        fromString.Update(data);

        using var fromLatin1 = Hasher.NewDeriveKey(Encoding.Latin1.GetBytes(NonAsciiContext));
        fromLatin1.Update(data);

        Assert.NotEqual(fromLatin1.Finalize().ToString(), fromString.Finalize().ToString());
    }

    [Fact]
    public void SubtreeContextDeriveKeyContextIsEncodedAsUtf8()
    {
        const long pieceSize = 1024;
        byte[] data = Input();

        using var fromString = Blake3SubtreeContext.CreateDeriveKey(NonAsciiContext, pieceSize);
        using var fromUtf8 = Blake3SubtreeContext.CreateDeriveKey(
            Encoding.UTF8.GetBytes(NonAsciiContext), pieceSize);

        Assert.Equal(WholeDigest(fromUtf8, data, pieceSize), WholeDigest(fromString, data, pieceSize));
    }

    /// <summary>
    /// The subtree context and the plain hasher must derive the same key from the same string,
    /// so the two independent call sites cannot drift apart.
    /// </summary>
    [Fact]
    public void SubtreeContextAndHasherAgreeOnTheSameContextString()
    {
        const long pieceSize = 1024;
        byte[] data = Input();

        using var context = Blake3SubtreeContext.CreateDeriveKey(NonAsciiContext, pieceSize);

        using var hasher = Hasher.NewDeriveKey(NonAsciiContext);
        hasher.Update(data);

        Assert.Equal(hasher.Finalize().ToString(), WholeDigest(context, data, pieceSize));
    }

    private static string WholeDigest(Blake3SubtreeContext context, byte[] data, long pieceSize)
    {
        var pieces = new List<Blake3Subtree>();
        for (int offset = 0, index = 0; offset < data.Length; offset += (int)pieceSize, index++)
        {
            int take = (int)Math.Min(pieceSize, data.Length - offset);
            pieces.Add(context.HashSubtree(data.AsSpan(offset, take), index));
        }

        return context.Finalize(pieces.ToArray()).ToString();
    }
}
