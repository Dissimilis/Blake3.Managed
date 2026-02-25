using System;

namespace Blake3.Managed.Tests;

public abstract class Blake3TestsBase
{
    protected static void AssertTextAreEqual(string expected, string result)
    {
        if (expected != result)
        {
            Console.WriteLine($"Expected: {expected}");
            Console.WriteLine($"  Result: {result}");
        }
        Assert.Equal(expected, result);
    }
}
