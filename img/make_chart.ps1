# Regenerates the README benchmark chart from BenchmarkDotNet results.
# Run: pwsh img/make_chart.ps1
Add-Type -AssemblyName System.Drawing

# ratio data per group, sorted ascending; (method, ratio)
$groups = [ordered]@{
    '4 Bytes' = @(
        @('Blake3 (native)', 1.00), @('Blake3.Managed', 1.07),
        @('SHA256', 1.94), @('Blake3.Managed (HashAlg)', 2.14))
    '100 Bytes' = @(
        @('Blake3 (native)', 1.00), @('Blake3.Managed', 1.04),
        @('SHA256', 1.61), @('Blake3.Managed (HashAlg)', 1.91))
    '1,000 Bytes' = @(
        @('SHA256', 0.64), @('Blake3 (native)', 1.00),
        @('Blake3.Managed', 1.05), @('Blake3.Managed (HashAlg)', 1.16))
    '10,000 Bytes' = @(
        @('Blake3 (native)', 1.00), @('Blake3.Managed', 1.33),
        @('Blake3.Managed (HashAlg)', 1.40), @('SHA256', 1.61))
    '100,000 Bytes' = @(
        @('Blake3 (native)', 1.00), @('Blake3.Managed', 1.47),
        @('Blake3.Managed (HashAlg)', 1.62), @('SHA256', 2.97))
    '1,000,000 Bytes' = @(
        @('Blake3.Managed', 0.42), @('Blake3 (native)', 1.00),
        @('Blake3.Managed (HashAlg)', 1.40), @('SHA256', 3.05))
}

$width = 756
$rowH = 45
$groupGap = 22
$headerH = 50
$rows = ($groups.Values | ForEach-Object { $_.Count } | Measure-Object -Sum).Sum
$height = $headerH + $rows * $rowH + ($groups.Count - 1) * $groupGap + 30

$bmp = New-Object System.Drawing.Bitmap($width, $height)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias

$bg        = [System.Drawing.Color]::FromArgb(24, 24, 24)
$txtDim    = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(200, 200, 200))
$txtBold   = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(235, 235, 235))
$barBrush  = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(205, 205, 205))
$trackBrush= New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(62, 62, 62))
$g.Clear($bg)

$fontHdr    = New-Object System.Drawing.Font('Segoe UI', 11, [System.Drawing.FontStyle]::Regular)
$fontSize   = New-Object System.Drawing.Font('Segoe UI', 11, [System.Drawing.FontStyle]::Bold)
$fontMethod = New-Object System.Drawing.Font('Segoe UI', 11, [System.Drawing.FontStyle]::Regular)
$fontNote   = New-Object System.Drawing.Font('Segoe UI', 10, [System.Drawing.FontStyle]::Italic)

$colSize = 14; $colMethod = 185; $colRatio = 430; $colBar = 500
$pxPerRatio = 52.0

# Header
$g.DrawString('Data Size', $fontHdr, $txtDim, $colSize, 14)
$g.DrawString('Method', $fontHdr, $txtDim, $colMethod, 14)
$g.DrawString('Ratio', $fontHdr, $txtDim, $colRatio, 14)
$g.DrawString('Visual Performance (vs Native)', $fontHdr, $txtDim, $colBar, 14)

$y = $headerH + 8
foreach ($entry in $groups.GetEnumerator()) {
    $first = $true
    foreach ($row in $entry.Value) {
        if ($first) { $g.DrawString($entry.Key, $fontSize, $txtBold, $colSize, $y + 11) }
        $method = $row[0]; $ratio = [double]$row[1]
        $g.DrawString($method, $fontMethod, $txtDim, $colMethod, $y + 11)
        $g.DrawString($ratio.ToString('0.00', [cultureinfo]::InvariantCulture), $fontMethod, $txtDim, $colRatio, $y + 11)

        $barLen = [int]($ratio * $pxPerRatio)
        $g.FillRectangle($trackBrush, $colBar, $y + 10, $barLen + 7, 24)
        $g.FillRectangle($barBrush, $colBar + 2, $y + 12, $barLen + 3, 20)
        if ($ratio -lt 1.0) {
            $g.DrawString('(Faster than baseline)', $fontNote, $txtDim, $colBar + $barLen + 16, $y + 13)
        }
        $y += $rowH
        $first = $false
    }
    $y += $groupGap
}

$g.Dispose()

$jpegCodec = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq 'image/jpeg' }
$encParams = New-Object System.Drawing.Imaging.EncoderParameters(1)
$encParams.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, 92L)
$out = Join-Path $PSScriptRoot 'benchmark_v0610.jpg'
$bmp.Save($out, $jpegCodec, $encParams)
$bmp.Dispose()
Write-Host "Saved $out"
