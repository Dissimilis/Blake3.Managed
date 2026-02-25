using System.Text.Json.Serialization;

namespace Blake3.Managed.Tests;

public class TestVectorFile
{
    [JsonPropertyName("key")]
    public string Key { get; set; } = "";

    [JsonPropertyName("context_string")]
    public string ContextString { get; set; } = "";

    [JsonPropertyName("cases")]
    public List<TestCase> Cases { get; set; } = new();
}

public class TestCase
{
    [JsonPropertyName("input_len")]
    public int InputLen { get; set; }

    [JsonPropertyName("hash")]
    public string Hash { get; set; } = "";

    [JsonPropertyName("keyed_hash")]
    public string KeyedHash { get; set; } = "";

    [JsonPropertyName("derive_key")]
    public string DeriveKey { get; set; } = "";
}
