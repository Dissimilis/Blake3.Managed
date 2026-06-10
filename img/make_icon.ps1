# Generates the NuGet package icon (img/icon.png, 128x128).
# Run: pwsh img/make_icon.ps1
Add-Type -AssemblyName System.Drawing

$size = 128
$bmp = New-Object System.Drawing.Bitmap($size, $size)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAlias
$g.Clear([System.Drawing.Color]::Transparent)

# Rounded dark square
$radius = 24
$path = New-Object System.Drawing.Drawing2D.GraphicsPath
$path.AddArc(0, 0, $radius * 2, $radius * 2, 180, 90)
$path.AddArc($size - $radius * 2, 0, $radius * 2, $radius * 2, 270, 90)
$path.AddArc($size - $radius * 2, $size - $radius * 2, $radius * 2, $radius * 2, 0, 90)
$path.AddArc(0, $size - $radius * 2, $radius * 2, $radius * 2, 90, 90)
$path.CloseFigure()
$bgBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(31, 36, 48))
$g.FillPath($bgBrush, $path)

# "B3" monogram: light B, blue 3
$font = New-Object System.Drawing.Font('Segoe UI', 52, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
$whiteBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(232, 234, 237))
$blueBrush  = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(59, 158, 255))

$sf = [System.Drawing.StringFormat]::GenericTypographic
$bSize = $g.MeasureString('B', $font, [System.Drawing.PointF]::Empty, $sf)
$tSize = $g.MeasureString('3', $font, [System.Drawing.PointF]::Empty, $sf)
$totalW = $bSize.Width + $tSize.Width
$x = ($size - $totalW) / 2
$y = ($size - $bSize.Height) / 2
$g.DrawString('B', $font, $whiteBrush, $x, $y, $sf)
$g.DrawString('3', $font, $blueBrush, $x + $bSize.Width, $y, $sf)

$g.Dispose()
$out = Join-Path $PSScriptRoot 'icon.png'
$bmp.Save($out, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()
Write-Host "Saved $out"
