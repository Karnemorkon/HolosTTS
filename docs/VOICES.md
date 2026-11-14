# Голосові паки

HolosTTS постачається з набором вбудованих голосів. Голосовий пак (voice
pack) — це окремий набір 256-вимірних ембеддінгів, який можна додати до
моделі або замінити ними вбудовані голоси. Паки не змінюють ваги моделі:
вони лише визначають, які голоси доступні через `list_voices()` і параметр
`voice=`.

## Навіщо це

Вбудовані голоси опубліковані без підтвердженої згоди їхніх власників (див.
розділ «License» у README). Для комерційного використання можна:

- узяти пак із чистою ліцензією, наприклад
  [`Morkon/holos-voices-uk-cc0`](https://huggingface.co/datasets/Morkon/holos-voices-uk-cc0)
  (29 українських голосів, CC0-1.0);
- клонувати власні голоси з аудіо, ліцензія якого дозволяє комерційне
  використання;
- змішати кілька голосів так, щоб жоден із них не був упізнаваний.

Пак не входить до ваг моделі; його ліцензія завжди окрема: перевіряйте
`LICENSE`, `SOURCES.md` і `SHA256SUMS` у самому паку.

## Формати

| Середовище | Файл | Формат |
| --- | --- | --- |
| PyTorch, `holos-tts` | `voices.pt` | `dict`: ім'я голосу → `float32[256]` |
| ONNX, `holos-tts-onnx` | `voices.npz` | рівно `names` (`str[N]`) і `values` (`float32[N, 256]`) |
| MLX, `holos-tts-mlx` | `voices.npz` | те саме, що для ONNX |
| LiteRT, `holos_litert` | `voices.bin` + `holos_litert.json` | сирий `float32[N, 256]` + імена й `dim` у маніфесті |
| legacy / інструменти | `voices.json` | `dict`: ім'я голосу → `[256 floats]` |

`holos-tts-onnx` вимагає, щоб `voices.npz` містив **рівно** два масиви:
`names` і `values`. Зайві ключі (`ids`, `groups`, `app_postprocess` тощо)
викликають помилку:

```text
ValueError: ... must contain exactly the 'names' and 'values' arrays
```

Метадані паку (описові назви, `id`, `legacy_id`, групи, походження) зберігайте
окремо — у `names.json`, `manifest.json` або `SOURCES.md`. Середовища
виконання читають лише `names` і `values`; решта файлів потрібна людям та
інструментам.

## Типовий пак

Мінімальний пак для ONNX/MLX — один файл `voices.npz`. Повний пак може
містити:

```text
voices.npz          # ONNX/MLX: names + values
voices.pt           # PyTorch: dict name -> float32[256]
voices.bin          # LiteRT: raw float32, C-order
voices.json         # legacy: dict name -> [256 floats]
names.json          # описові назви, id, legacy_id, групи
manifest.json       # метрики, походження, app_postprocess
SOURCES.md          # точні кліпи, client_id, ліцензії
LICENSE             # ліцензія паку
SHA256SUMS          # контрольні суми
ref/*.wav           # референси, якщо є
samples/*.wav       # приклади синтезу, якщо є
preview/index.html  # сторінка прослуховування, якщо є
```

Не всі файли обов'язкові. Для заміни голосів у конкретному середовищі
достатньо файлу з таблиці вище.

## Встановлення

### ONNX і MLX

1. Завантажте локальну копію бандла, наприклад:

   ```bash
   huggingface-cli download patriotyk/HolosTTS \
     --include "onnx/*" --local-dir holos-onnx
   ```

2. Зробіть резервну копію `voices.npz` і замініть його файлом паку:

   ```bash
   cp holos-onnx/voices.npz holos-onnx/voices.npz.orig
   cp holos-voices-uk-cc0/voices.npz holos-onnx/voices.npz
   ```

3. Передайте локальний каталог у конструктор:

   ```python
   from holos_onnx import HolosTTS

   model = HolosTTS("holos-onnx")
   print(model.list_voices())
   ```

   Для MLX — те саме з `holos_mlx` і каталогом `mlx/`.

Якщо потрібно зберегти вбудовані голоси й додати пак, використайте
`scripts/merge_voice_pack.py` (див. нижче). Він створює `voices.npz` із
двома масивами, сумісний з обома середовищами.

### PyTorch

```python
from holos.models import HolosTTS

model = HolosTTS(
    "patriotyk/HolosTTS",
    voices_path="holos-voices-uk-cc0/voices.pt",
)
print(model.list_voices())
```

`voices_path` може бути локальним файлом; ваги моделі при цьому можна
завантажувати з Hugging Face. Якщо `voices.pt` лежить поруч із локальним
`holos.pt`, достатньо `HolosTTS("holos.pt", voices_path="voices.pt")`.

### LiteRT

LiteRT читає імена з `holos_litert.json`, а ембеддінги — із `voices.bin`.
Потрібно замінити обидва файли:

```bash
cp holos-litert/holos_litert.json holos-litert/holos_litert.json.orig
cp holos-litert/voices.bin holos-litert/voices.bin.orig
cp holos-litert-cc0/holos_litert.json holos-litert/holos_litert.json
cp holos-litert-cc0/voices.bin holos-litert/voices.bin
```

Секція `voices` у `holos_litert.json` має вигляд:

```json
{
  "voices": {
    "file": "voices.bin",
    "names": ["ім'я 1", "ім'я 2"],
    "dim": 256
  }
}
```

Решта полів маніфесту (`format`, `vocab`, `encoder`, `prosody`, `decoder`,
`embeddings`, `stft` тощо) залишається без змін.

Кількість імен має дорівнювати `розмір voices.bin / (4 * 256)`. Якщо
розмір не збігається, LiteRT не завантажиться.

## Злиття паку з наявними голосами

`scripts/merge_voice_pack.py` додає голоси паку до голосів бандла або
замінює їх. Скрипт перевіряє розмірність, `NaN`/`Inf`, порожні та повторювані
імена, читає ціль повністю перед записом і не перезаписує файли без
`--force`.

### Додати пак, зберігши вбудовані голоси

```bash
python scripts/merge_voice_pack.py \
  holos-voices-uk-cc0 \
  holos-onnx \
  --output holos-onnx/voices-cc0.npz \
  --prefix "cc0/"
```

Префікс `cc0/` гарантує, що імена не збігаються з вбудованими. Потім
замініть `voices.npz` у бандлі:

```bash
cp holos-onnx/voices-cc0.npz holos-onnx/voices.npz
```

Можна писати одразу в цільовий файл, попередньо зробивши резервну копію:

```bash
python scripts/merge_voice_pack.py \
  holos-voices-uk-cc0 holos-onnx \
  --output holos-onnx/voices.npz \
  --prefix "cc0/" --force
```

### Замінити всі голоси

```bash
python scripts/merge_voice_pack.py \
  holos-voices-uk-cc0 holos-onnx \
  --output holos-onnx/voices.npz \
  --replace --force
```

Для PyTorch:

```bash
python scripts/merge_voice_pack.py \
  holos-voices-uk-cc0 holos/voices.pt \
  --output holos/voices-cc0.pt \
  --replace
```

Для LiteRT:

```bash
python scripts/merge_voice_pack.py \
  holos-voices-uk-cc0 holos-litert \
  --output holos-litert-cc0 \
  --format litert --prefix "cc0/"
```

Скрипт створить `holos-litert-cc0/voices.bin` і
`holos-litert-cc0/holos_litert.json`. Скопіюйте обидва файли в бандл, як
показано вище.

### Корисні параметри

| Параметр | Дія |
| --- | --- |
| `--replace` | використати лише голоси паку, без вбудованих |
| `--prefix "cc0/"` | додати префікс до імен паку, щоб уникнути колізій |
| `--skip-duplicates` | пропустити голоси, чиї імена вже є в цілі |
| `--format npz\|pt\|json\|litert` | явно вибрати формат виходу |
| `--force` | дозволити перезапис існуючих файлів |
| `--dry-run` | показати план і нічого не записувати |

Якщо `--format` не заданий, формат визначається за розширенням вихідного
файлу (`.npz`, `.pt`, `.json`), а для каталогу — за форматом цілі.

### Імена та колізії

Середовища виконання використовують ім'я голосу як ключ. Два голоси з
однаковим іменем неможливі: `list_voices()` і `voice=` працюють із рядком.
Якщо пак додається до бандла, де вже є голос із таким іменем, скрипт
зупиниться й запропонує `--prefix`, `--skip-duplicates` або `--replace`.

`names.json` і `manifest.json` можуть містити додаткові `id` та `legacy_id`,
але середовища виконання їх не читають. Для сумісності змінюйте саме
`names`/`values`, а не метадані.

## Перевірка паку

Мінімальна перевірка `voices.npz`:

```python
import numpy as np

with np.load("voices.npz", allow_pickle=False) as archive:
    assert set(archive.files) == {"names", "values"}, archive.files
    names = archive["names"].astype(str).tolist()
    values = archive["values"]

assert values.shape == (len(names), 256), values.shape
assert values.dtype == np.float32, values.dtype
assert np.isfinite(values).all()
assert len(set(names)) == len(names)
print(f"{len(names)} voices, {values.shape[1]} dimensions")
```

Контрольні суми:

```bash
cd holos-voices-uk-cc0
sha256sum -c SHA256SUMS
```

`SHA256SUMS` не містить самого себе; це нормально.

## Ліцензії та походження

- **Модель HolosTTS** — MIT. Ваги моделі не входять до голосового паку.
- **Голосовий пак** — окрема ліцензія. CC0-пак
  `Morkon/holos-voices-uk-cc0` поширюється за CC0-1.0: аудіо, ембеддінги,
  метадані та референси.
- **Походження CC0-паку** — Mozilla Common Voice Scripted Speech 27.0
  (Ukrainian). Точні `client_id` і кліпи — у `SOURCES.md`. Ідентифікація
  дикторів не виконувалася.
- **Вбудовані голоси HolosTTS** не мають підтвердженої згоди власників.
  Не поширюйте їх як частину комерційного продукту без окремого дозволу.

Якщо пак містить `manifest.json` з `app_postprocess`, середовище виконання
не застосовує його автоматично. Наприклад, у CC0-паку голос
`child-girl-high-warm` (`cv_uk_k33`) зберігається raw, а застосунок може
додати `highpass=f=70,rubberband=tempo=1.05:formant=preserved`. Це
відповідальність застосунку або проксі, а не моделі.

## Типові помилки

| Помилка | Причина | Рішення |
| --- | --- | --- |
| `must contain exactly the 'names' and 'values' arrays` | у `voices.npz` є зайві ключі | перепакуйте з двома масивами або використайте скрипт |
| `Unknown voice '...'` | імені немає в `names` | `print(model.list_voices())`, перевірте регістр і префікс |
| `reading voices.pt needs PyTorch` | для `.pt` потрібен PyTorch | використайте `voices.npz`/`voices.json` або встановіть PyTorch |
| `expected 256-dimensional voices` | неправильна розмірність | пак має містити 256-вимірні ембеддінги HolosTTS |
| LiteRT: кількість імен не збігається з `voices.bin` | оновлено лише один файл | оновіть і `holos_litert.json`, і `voices.bin` |
| `voice names already exist` | колізія з вбудованими голосами | `--prefix`, `--skip-duplicates` або `--replace` |
