#Requires -Version 5.1
<#
    GlassTranslate model installer

    Downloads the models that are not bundled with the portable zip, straight
    from Hugging Face, into the folder your copy of GlassTranslate uses.

    Just run it with no arguments for the interactive flow.

      -Path <folder>   skip the folder picker
      -Only <ids>      skip the menu (mangaocr, quality, sugoi)
      -DryRun          show what would be downloaded, download nothing
#>
[CmdletBinding()]
param(
    [string]   $Path,
    [string[]] $Only,
    [switch]   $DryRun
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'   # we draw our own bar

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$CATALOG = @'
{
 "groups": [
  {
   "id": "mangaocr",
   "name": "manga-ocr (Japanese OCR)",
   "note": "Reads the Japanese text. Install this one.",
   "files": [
    {
     "url": "https://huggingface.co/onnx-community/manga-ocr-base-ONNX/resolve/f9023406bb2f6b17df67bc4a327c56ecd20611f0/onnx/encoder_model_fp16.onnx",
     "rel": "manga-ocr/encoder_model_fp16.onnx",
     "size": 171851891,
     "sha256": "1a6a57bc3608195c4577b13ac3aadab810dce42fa22c5a3acf0570bffc013b60"
    },
    {
     "url": "https://huggingface.co/onnx-community/manga-ocr-base-ONNX/resolve/f9023406bb2f6b17df67bc4a327c56ecd20611f0/onnx/decoder_model_int8.onnx",
     "rel": "manga-ocr/decoder_model_int8.onnx",
     "size": 29627936,
     "sha256": "2e7177d2b0a59f1c612b694ed70c13971bee765cc2b2bc7bc9376e4753652f27"
    },
    {
     "url": "https://huggingface.co/onnx-community/manga-ocr-base-ONNX/resolve/f9023406bb2f6b17df67bc4a327c56ecd20611f0/config.json",
     "rel": "manga-ocr/config.json",
     "size": 75028,
     "sha256": "0b45d5253cef67122d2d35b32408ccffa406a87d560e2c3b7ddec3a987863f2e"
    },
    {
     "url": "https://huggingface.co/onnx-community/manga-ocr-base-ONNX/resolve/f9023406bb2f6b17df67bc4a327c56ecd20611f0/generation_config.json",
     "rel": "manga-ocr/generation_config.json",
     "size": 261,
     "sha256": "faa07a187c72b147a39f660993bcbdbb85c5fc1a86f8a557d5487945285648c0"
    },
    {
     "url": "https://huggingface.co/onnx-community/manga-ocr-base-ONNX/resolve/f9023406bb2f6b17df67bc4a327c56ecd20611f0/preprocessor_config.json",
     "rel": "manga-ocr/preprocessor_config.json",
     "size": 351,
     "sha256": "445c77049d082004aa07593344d5e1f521f1198228bf586196f70ce7ae021414"
    },
    {
     "url": "https://huggingface.co/kha-white/manga-ocr-base/resolve/aa6573bd10b0d446cbf622e29c3e084914df9741/vocab.txt",
     "rel": "manga-ocr/vocab.txt",
     "size": 24072,
     "sha256": "344fbb6b8bf18c57839e924e2c9365434697e0227fac00b88bb4899b78aa594d"
    },
    {
     "url": "https://huggingface.co/kha-white/manga-ocr-base/resolve/aa6573bd10b0d446cbf622e29c3e084914df9741/tokenizer_config.json",
     "rel": "manga-ocr/tokenizer_config.json",
     "size": 486,
     "sha256": "d775ad1deac162dc56b84e9b8638f95ed8a1f263d0f56f4f40834e26e205e266"
    }
   ]
  },
  {
   "id": "quality",
   "name": "Quality renderer",
   "note": "Redraws artwork behind erased text. Needs an NVIDIA GPU.",
   "files": [
    {
     "url": "https://huggingface.co/TareHimself/AnimeMangaInpainting-torchscript/resolve/b592884b2b6589e57df6916db6bcac7a6307fc58/anime_manga_lama.pt",
     "rel": "quality/lama/anime_manga_lama.pt",
     "size": 204190262,
     "sha256": "95d4bf7eb2f13729351ac0a0693c9b9b469b264fc1542ca3fa551b59b464a160"
    },
    {
     "url": "https://huggingface.co/TareHimself/AnimeMangaInpainting-torchscript/resolve/b592884b2b6589e57df6916db6bcac7a6307fc58/config.json",
     "rel": "quality/lama/config.json",
     "size": 1480,
     "sha256": "d77a2f1631bde802d547ef61bd98ed48fcccf97c8bb7d652f7c1def5cbf9e47e"
    },
    {
     "url": "https://huggingface.co/TareHimself/AnimeMangaInpainting-torchscript/resolve/b592884b2b6589e57df6916db6bcac7a6307fc58/LICENSE",
     "rel": "quality/lama/LICENSE",
     "size": 1308,
     "sha256": "2f40440bfd9c728e72a5356b64d491724d285db30cb1b8be2f0fcb441880593d"
    },
    {
     "url": "https://huggingface.co/TareHimself/AnimeMangaInpainting-torchscript/resolve/b592884b2b6589e57df6916db6bcac7a6307fc58/NOTICE",
     "rel": "quality/lama/NOTICE",
     "size": 882,
     "sha256": "95a66aceef0b28aa69ec013312e7a6a42c654edfae9a04b6cf0902083a6152d2"
    },
    {
     "url": "https://huggingface.co/OnomaAIResearch/Illustrious-XL-v1.0/resolve/89d625482edf06d545b740265482f5c8fb2cadb0/Illustrious-XL-v1.0.safetensors",
     "rel": "quality/sdxl/Illustrious-XL-v1.0.safetensors",
     "size": 6938040736,
     "sha256": "735cf3fefcbdc4f7817f53247e38b836ffd27c7641af6d8daa21d245242cb4bd"
    },
    {
     "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/LICENSE.md",
     "rel": "quality/sdxl/LICENSE.md",
     "size": 14109,
     "sha256": "19b6998b569b53ac1fc2158a8a3202c8699a9a4605b47075715d9c96be7fb6d0"
    },
    {
     "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/model_index.json",
     "rel": "quality/sdxl-base-config/model_index.json",
     "size": 609,
     "sha256": "6d7b93508390ab91ac5bfbe4aeb4dc2d83f7bb1b05fb069d714b5b0c75f70d44"
    },
    {
     "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/scheduler/scheduler_config.json",
     "rel": "quality/sdxl-base-config/scheduler/scheduler_config.json",
     "size": 479,
     "sha256": "af3e45a949aff8b8341ab8b811429ec03fee857a700a1d9477363e4fff9666e2"
    },
    {
     "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/text_encoder/config.json",
     "rel": "quality/sdxl-base-config/text_encoder/config.json",
     "size": 565,
     "sha256": "39b8b2e4b1949e36969caa425b6c81c68bace99198dd9078ce05d16ad401fe7f"
    },
    {
     "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/text_encoder_2/config.json",
     "rel": "quality/sdxl-base-config/text_encoder_2/config.json",
     "size": 575,
     "sha256": "a892d1c3a69a7e9247a24de2bc1d5891e3109a54696e53be20093af671072c34"
    },
    {
     "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/tokenizer/vocab.json",
     "rel": "quality/sdxl-base-config/tokenizer/vocab.json",
     "size": 1059962,
     "sha256": "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"
    },
    {
     "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/tokenizer/merges.txt",
     "rel": "quality/sdxl-base-config/tokenizer/merges.txt",
     "size": 524619,
     "sha256": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"
    },
    {
     "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/tokenizer/tokenizer_config.json",
     "rel": "quality/sdxl-base-config/tokenizer/tokenizer_config.json",
     "size": 737,
     "sha256": "19d7b034cb0cc3ce9766c2231373ab8aa8991fc72e2c8f76558bfaae3de0d563"
    },
    {
     "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/tokenizer/special_tokens_map.json",
     "rel": "quality/sdxl-base-config/tokenizer/special_tokens_map.json",
     "size": 472,
     "sha256": "c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac"
    },
    {
     "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/tokenizer_2/vocab.json",
     "rel": "quality/sdxl-base-config/tokenizer_2/vocab.json",
     "size": 1059962,
     "sha256": "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"
    },
    {
     "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/tokenizer_2/merges.txt",
     "rel": "quality/sdxl-base-config/tokenizer_2/merges.txt",
     "size": 524619,
     "sha256": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"
    },
    {
     "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/tokenizer_2/tokenizer_config.json",
     "rel": "quality/sdxl-base-config/tokenizer_2/tokenizer_config.json",
     "size": 725,
     "sha256": "c9d23941f76a41cbd50eda9290f57be7828f0a7a677939e9ef181f7e12bd1bdf"
    },
    {
     "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/tokenizer_2/special_tokens_map.json",
     "rel": "quality/sdxl-base-config/tokenizer_2/special_tokens_map.json",
     "size": 460,
     "sha256": "f118ab3a983206e4f32583448de6bd6aae4ee21869135cef1f5848a753cdaab6"
    },
    {
     "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/unet/config.json",
     "rel": "quality/sdxl-base-config/unet/config.json",
     "size": 1680,
     "sha256": "30ebc70750223e59006f7f2b4e1e6c102570aa19a9c4ae3e1fbe7591332dbae6"
    },
    {
     "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/vae/config.json",
     "rel": "quality/sdxl-base-config/vae/config.json",
     "size": 642,
     "sha256": "0b331c8ac22ded5f9997a144a575c1d113d6169aff262c353f39015bd24a6264"
    },
    {
     "url": "https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/801a4a3fa3d4c936f4feea95b98607bc6726f80c/diffusion_pytorch_model_promax.safetensors",
     "rel": "quality/controlnet/diffusion_pytorch_model.safetensors",
     "size": 2513342408,
     "sha256": "9fae2e50cb431bfcbe05822b59ec2228df545ef27f711dea8949e9f4ed9f7cdc"
    },
    {
     "url": "https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/801a4a3fa3d4c936f4feea95b98607bc6726f80c/config_promax.json",
     "rel": "quality/controlnet/config.json",
     "size": 1260,
     "sha256": "6653ad6a0ed181f0d4a7225f1b1c037405d98573fbf0862b1ca3ab6c99b52c21"
    },
    {
     "url": "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/207b116dae70ace3637169f1ddd2434b91b3a8cd/diffusion_pytorch_model.safetensors",
     "rel": "quality/vae/diffusion_pytorch_model.safetensors",
     "size": 334643238,
     "sha256": "1b909373b28f2137098b0fd9dbc6f97f8410854f31f84ddc9fa04b077b0ace2c"
    },
    {
     "url": "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/207b116dae70ace3637169f1ddd2434b91b3a8cd/config.json",
     "rel": "quality/vae/config.json",
     "size": 631,
     "sha256": "f18b16fa4381c90ab44a3d7abd2e90afd05a2c31b8ca69998bc93c6453aeb7b7"
    }
   ]
  },
  {
   "id": "sugoi",
   "name": "Sugoi v4 (Japanese -> English)",
   "note": "Far better manga translation than the default. Research licence.",
   "files": [
    {
     "url": "https://huggingface.co/entai2965/sugoi-v4-ja-en-ctranslate2/resolve/main/config.json",
     "rel": "sugoi-v4-ja-en/config.json",
     "size": 0,
     "sha256": ""
    },
    {
     "url": "https://huggingface.co/entai2965/sugoi-v4-ja-en-ctranslate2/resolve/main/model.bin",
     "rel": "sugoi-v4-ja-en/model.bin",
     "size": 0,
     "sha256": ""
    },
    {
     "url": "https://huggingface.co/entai2965/sugoi-v4-ja-en-ctranslate2/resolve/main/source_vocabulary.json",
     "rel": "sugoi-v4-ja-en/source_vocabulary.json",
     "size": 0,
     "sha256": ""
    },
    {
     "url": "https://huggingface.co/entai2965/sugoi-v4-ja-en-ctranslate2/resolve/main/target_vocabulary.json",
     "rel": "sugoi-v4-ja-en/target_vocabulary.json",
     "size": 0,
     "sha256": ""
    },
    {
     "url": "https://huggingface.co/entai2965/sugoi-v4-ja-en-ctranslate2/resolve/main/spm/spm.ja.nopretok.model",
     "rel": "sugoi-v4-ja-en/spm/spm.ja.nopretok.model",
     "size": 0,
     "sha256": ""
    },
    {
     "url": "https://huggingface.co/entai2965/sugoi-v4-ja-en-ctranslate2/resolve/main/spm/spm.en.nopretok.model",
     "rel": "sugoi-v4-ja-en/spm/spm.en.nopretok.model",
     "size": 0,
     "sha256": ""
    }
   ],
   "metadata": {
    "rel": "sugoi-v4-ja-en/metadata.json",
    "body": {
     "from_code": "ja",
     "to_code": "en",
     "from_name": "Japanese",
     "to_name": "English",
     "source_spm": "spm/spm.ja.nopretok.model",
     "target_spm": "spm/spm.en.nopretok.model",
     "priority": 10,
     "source": "https://huggingface.co/entai2965/sugoi-v4-ja-en-ctranslate2/resolve/main"
    }
   }
  }
 ]
}
'@ | ConvertFrom-Json

# ---------------------------------------------------------------- helpers ---

function Format-Size([double] $b) {
    if ($b -ge 1GB) { return ('{0:N2} GB' -f ($b / 1GB)) }
    if ($b -ge 1MB) { return ('{0:N1} MB' -f ($b / 1MB)) }
    if ($b -ge 1KB) { return ('{0:N0} KB' -f ($b / 1KB)) }
    return "$([int]$b) B"
}

function Format-Span([double] $sec) {
    if ($sec -lt 0 -or [double]::IsInfinity($sec) -or [double]::IsNaN($sec)) { return '--:--' }
    $t = [TimeSpan]::FromSeconds([Math]::Min($sec, 359999))
    if ($t.TotalHours -ge 1) { return ('{0}:{1:00}:{2:00}' -f [int]$t.TotalHours, $t.Minutes, $t.Seconds) }
    return ('{0:00}:{1:00}' -f $t.Minutes, $t.Seconds)
}

function Write-Bar {
    param([double] $Done, [double] $Total, [double] $Rate)

    $width = 30
    $frac  = if ($Total -gt 0) { [Math]::Min(1.0, $Done / $Total) } else { 0 }
    $fill  = [int][Math]::Floor($width * $frac)
    $bar   = ([string][char]0x2588) * $fill + ([string][char]0x2591) * ($width - $fill)
    $eta   = if ($Rate -gt 0 -and $Total -gt 0) { Format-Span (($Total - $Done) / $Rate) } else { '--:--' }

    $line = '  [{0}] {1,3:N0}%  {2} / {3}  {4}/s  ETA {5}' -f `
            $bar, ($frac * 100), (Format-Size $Done), (Format-Size $Total), (Format-Size $Rate), $eta

    $w = 100
    try { $w = $Host.UI.RawUI.WindowSize.Width } catch { }
    $pad = [Math]::Max(0, ($w - 1) - $line.Length)
    Write-Host ("`r" + $line + (' ' * $pad)) -NoNewline -ForegroundColor Cyan
}

function Get-Sha256([string] $File) {
    (Get-FileHash -Path $File -Algorithm SHA256).Hash.ToLower()
}

# Streams $Url to $Dest with a live bar, resuming a previous .part if present,
# and hashing as it goes so the file is never re-read just to verify it.
function Get-Remote {
    param([string] $Url, [string] $Dest, [long] $Size, [string] $Sha, [string] $Label)

    $dir = Split-Path -Parent $Dest
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

    if (Test-Path $Dest) {
        $len    = (Get-Item $Dest).Length
        $sizeOk = ($Size -le 0) -or ($len -eq $Size)
        if ($sizeOk -and $Sha) {
            if ((Get-Sha256 $Dest) -eq $Sha.ToLower()) {
                Write-Host "  [skip] $Label (already verified)" -ForegroundColor DarkGray
                return
            }
        } elseif ($sizeOk -and $len -gt 0) {
            Write-Host "  [skip] $Label (already present)" -ForegroundColor DarkGray
            return
        }
        Remove-Item $Dest -Force
    }

    $part   = "$Dest.part"
    $have   = 0L
    $hasher = [Security.Cryptography.IncrementalHash]::CreateHash([Security.Cryptography.HashAlgorithmName]::SHA256)

    # Re-hash whatever a previous run already fetched, then ask for the rest.
    if (Test-Path $part) {
        $have = (Get-Item $part).Length
        if ($have -gt 0) {
            $fs  = [IO.File]::OpenRead($part)
            $buf = New-Object byte[] (1MB)
            while (($n = $fs.Read($buf, 0, $buf.Length)) -gt 0) { $hasher.AppendData($buf, 0, $n) }
            $fs.Close()
            Write-Host ("  resuming at {0}" -f (Format-Size $have)) -ForegroundColor DarkGray
        }
    }

    $req = [Net.HttpWebRequest]::Create($Url)
    $req.UserAgent         = 'GlassTranslate-model-installer/1.0'
    $req.Timeout           = 60000
    $req.ReadWriteTimeout  = 120000
    $req.AllowAutoRedirect = $true
    if ($have -gt 0) { $req.AddRange($have) }

    try {
        $resp = $req.GetResponse()
    } catch [Net.WebException] {
        # 416 = the server says we already hold the whole file.
        $code = $null
        if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
        if ($code -eq 416 -and $have -gt 0) { Move-Item $part $Dest -Force; return }
        throw "download failed for ${Label}: $($_.Exception.Message)"
    }

    $total = $have + $resp.ContentLength
    if ($Size -gt 0) { $total = $Size }
    $in    = $resp.GetResponseStream()
    $out   = [IO.File]::Open($part, [IO.FileMode]::Append, [IO.FileAccess]::Write)
    $buf   = New-Object byte[] (1MB)
    $done  = $have
    $sw    = [Diagnostics.Stopwatch]::StartNew()
    $lastT = 0.0
    $lastB = $have
    $rate  = 0.0

    try {
        while (($n = $in.Read($buf, 0, $buf.Length)) -gt 0) {
            $out.Write($buf, 0, $n)
            $hasher.AppendData($buf, 0, $n)
            $done += $n
            $el = $sw.Elapsed.TotalSeconds
            if (($el - $lastT) -ge 0.25) {
                $rate  = ($done - $lastB) / ($el - $lastT)
                $lastT = $el
                $lastB = $done
                Write-Bar -Done $done -Total $total -Rate $rate
            }
        }
    } finally {
        $out.Close(); $in.Close(); $resp.Close()
    }

    Write-Bar -Done $done -Total $total -Rate $rate
    Write-Host ''

    $actual = [BitConverter]::ToString($hasher.GetHashAndReset()).Replace('-', '').ToLower()
    if ($Sha -and $actual -ne $Sha.ToLower()) {
        Remove-Item $part -Force -ErrorAction SilentlyContinue
        throw "checksum mismatch for ${Label} (expected $Sha, got $actual)"
    }
    if ($Size -gt 0 -and $done -ne $Size) {
        Remove-Item $part -Force -ErrorAction SilentlyContinue
        throw "size mismatch for ${Label} (expected $Size bytes, got $done)"
    }

    Move-Item $part $Dest -Force
}

# ------------------------------------------------------------ folder pick ---

function Select-Folder {
    Add-Type -AssemblyName System.Windows.Forms | Out-Null
    $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
    $dlg.Description         = 'Select your unpacked GlassTranslate folder (the one with GlassTranslate.exe)'
    $dlg.ShowNewFolderButton = $false
    if ($dlg.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { return $null }
    return $dlg.SelectedPath
}

function Resolve-ModelsDir([string] $Root) {
    if (-not (Test-Path (Join-Path $Root 'GlassTranslate.exe'))) {
        throw "That folder has no GlassTranslate.exe in it: $Root"
    }
    if (Test-Path (Join-Path $Root 'portable.txt')) {
        return (Join-Path $Root 'models')
    }
    # No portable marker: the app keeps its store in the user profile instead.
    return (Join-Path $env:LOCALAPPDATA 'GlassTranslate\models')
}

function Get-GroupSize($Group) {
    $s = ($Group.files | Measure-Object -Property size -Sum).Sum
    if (-not $s -or $s -le 0) { return 700MB }   # Sugoi has no pinned sizes
    return $s
}

# ------------------------------------------------------------------- menu ---

function Select-Groups($Groups) {
    $chosen = @{}
    foreach ($g in $Groups) { $chosen[$g.id] = ($g.id -eq 'mangaocr') }

    while ($true) {
        Clear-Host
        Write-Host ''
        Write-Host '  GlassTranslate - model installer' -ForegroundColor White
        Write-Host '  --------------------------------' -ForegroundColor DarkGray
        Write-Host ''
        for ($i = 0; $i -lt $Groups.Count; $i++) {
            $g    = $Groups[$i]
            $mark = if ($chosen[$g.id]) { 'x' } else { ' ' }
            $col  = if ($chosen[$g.id]) { 'Green' } else { 'Gray' }
            Write-Host ('   [{0}] {1}. {2,-32} {3,10}' -f $mark, ($i + 1), $g.name, (Format-Size (Get-GroupSize $g))) -ForegroundColor $col
            Write-Host ('          {0}' -f $g.note) -ForegroundColor DarkGray
        }
        $totalSel = 0
        foreach ($g in $Groups) { if ($chosen[$g.id]) { $totalSel += Get-GroupSize $g } }
        Write-Host ''
        Write-Host ('   Selected: {0}' -f (Format-Size $totalSel)) -ForegroundColor Yellow
        Write-Host ''
        Write-Host '   1-3 toggle    A all    N none    Enter install    Q quit' -ForegroundColor DarkGray
        Write-Host ''
        $key = Read-Host '  >'

        switch -Regex ($key) {
            '^\s*$'   { return @($Groups | Where-Object { $chosen[$_.id] }) }
            '^[Qq]'   { return @() }
            '^[Aa]'   { foreach ($g in $Groups) { $chosen[$g.id] = $true } }
            '^[Nn]'   { foreach ($g in $Groups) { $chosen[$g.id] = $false } }
            '^[1-9]$' {
                $idx = [int]$key - 1
                if ($idx -lt $Groups.Count) { $chosen[$Groups[$idx].id] = -not $chosen[$Groups[$idx].id] }
            }
        }
    }
}

# ------------------------------------------------------------------- main ---

$root = $Path
if (-not $root) { $root = Select-Folder }
if (-not $root) { Write-Host 'Cancelled.' -ForegroundColor Yellow; exit 1 }

$modelsDir = Resolve-ModelsDir $root
$groups    = @($CATALOG.groups)

if ($Only) {
    $want   = $Only | ForEach-Object { $_.ToLower() }
    $picked = @($groups | Where-Object { $want -contains $_.id })
    if (-not $picked) { throw "None of those ids exist. Valid: $(($groups | ForEach-Object { $_.id }) -join ', ')" }
} else {
    $picked = @(Select-Groups $groups)
}

if (-not $picked) { Write-Host 'Nothing selected.' -ForegroundColor Yellow; exit 0 }

Write-Host ''
Write-Host "  Installing into: $modelsDir" -ForegroundColor White
if (-not (Test-Path (Join-Path $root 'portable.txt'))) {
    Write-Host '  (no portable.txt here, so this copy keeps its models in your user profile)' -ForegroundColor DarkYellow
}
Write-Host ''

$allFiles = @()
foreach ($g in $picked) { foreach ($f in $g.files) { $allFiles += , $f } }

if ($DryRun) {
    foreach ($f in $allFiles) {
        '{0,-54} {1,10}  {2}' -f $f.rel, (Format-Size $f.size), $f.url
    }
    $sum = 0
    foreach ($g in $picked) { $sum += Get-GroupSize $g }
    Write-Host ''
    Write-Host ('  {0} files, {1}' -f $allFiles.Count, (Format-Size $sum)) -ForegroundColor Yellow
    Write-Host '  (dry run - nothing downloaded)' -ForegroundColor Yellow
    exit 0
}

$i  = 0
$t0 = Get-Date
foreach ($f in $allFiles) {
    $i++
    Write-Host ('  [{0}/{1}] {2}' -f $i, $allFiles.Count, $f.rel) -ForegroundColor White
    Get-Remote -Url $f.url -Dest (Join-Path $modelsDir $f.rel) -Size $f.size -Sha $f.sha256 -Label (Split-Path -Leaf $f.rel)
}

# Packs that need a descriptor before the app will register them.
foreach ($g in $picked) {
    if (($g.PSObject.Properties.Name -contains 'metadata') -and $g.metadata) {
        $dest = Join-Path $modelsDir $g.metadata.rel
        $dir  = Split-Path -Parent $dest
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        $g.metadata.body | ConvertTo-Json -Depth 5 | Set-Content -Path $dest -Encoding UTF8
        Write-Host "  wrote $($g.metadata.rel)" -ForegroundColor DarkGray
    }
}

Write-Host ''
Write-Host ('  Done in {0}. Start GlassTranslate.exe.' -f (Format-Span ((Get-Date) - $t0).TotalSeconds)) -ForegroundColor Green
Write-Host ''
