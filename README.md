# Robust Simplified MoE-DCS Project

این پروژه نسخه‌ی بازطراحی‌شده‌ی پروژه‌ی ساده‌شده‌ی مقاله‌ی **Robust Exploration in Directed Controller Synthesis via Mixture-of-Experts Reinforcement Learning** است.

## اصلاحات اصلی نسبت به نسخه اول

1. هر Expert روی یک **توزیع از محیط‌ها** آموزش می‌بیند، نه فقط یک Grid بسیار کوچک.
2. متخصص‌ها برای سه رژیم ساختاری ساخته شده‌اند: `wide`، `tall` و `balanced`.
3. State شامل موقعیت نرمال‌شده، Action Mask محلی و فاز جست‌وجو است.
4. محیط دارای مسیر امن تضمینی، شاخه‌های انحرافی و بن‌بست‌های پارامتری است.
5. Prior روی Seedهای کالیبراسیون ساخته می‌شود و آزمون نهایی روی Seedهای مستقل انجام می‌شود.
6. Prior از **موفقیت به‌عنوان معیار اصلی** و Efficiency برای شکستن تساوی استفاده می‌کند.
7. سه نوع MoE مقایسه می‌شوند: Sparse Top-1، Top-2، Top-3 و All Experts.
8. ترکیب Expertها با `log opinion pool` انجام می‌شود تا Expert ضعیف خروجی Expert قوی را بیش از حد محو نکند.
9. معیارهای موفقیت و Expansion اجرای موفق/ناموفق جدا ثبت می‌شوند.
10. Oracle Best Expert به‌عنوان سقف نظری Complementarity گزارش می‌شود.
11. Demo از Prior واقعی ذخیره‌شده استفاده می‌کند.
12. تنظیمات کامل آزمایش در `experiment_config.json` ذخیره می‌شوند.

## نصب

```bash
python -m venv .venv
```

ویندوز:

```powershell
.venv\Scripts\activate
```

لینوکس:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## اجرای سریع برای کنترل

```bash
python run_experiment.py --output quick_results --episodes 400 --calibration-max-n 10 --calibration-max-k 10 --test-min 3 --test-max 10 --budget 20 --eval-seeds 5
```

## اجرای اصلی پیشنهادی

```bash
python run_experiment.py --output results_v2 --episodes 1600 --calibration-max-n 14 --calibration-max-k 14 --test-min 3 --test-max 14 --budget 25 --eval-seeds 10
```

## اجرای Demo

```bash
python demo.py
```

## خروجی‌ها

- `evaluation.csv`: نتیجه هر Policy روی هر `(n,k)`
- `summary.csv`: خلاصه نهایی Policyها
- `prior_strengths.json`: نقشه قدرت کالیبراسیون
- `experiment_config.json`: تنظیمات دقیق آزمایش
- `heatmap_*.png`: Heatmap سیاست‌ها
- `policy_comparison.png`: مقایسه پوشش
- `models/*.json`: Q-table متخصص‌ها

## تفسیر معیارها

- `success_rate`: نسبت اجراهای موفق روی Seedهای آزمون
- `success_sum`: مجموع نرخ موفقیت تمام نقاط پارامتری
- `mean_expansions_success`: میانگین Expansion فقط در اجراهای موفق
- `mean_expansions_failure`: میانگین Expansion اجراهای شکست‌خورده
- `oracle_best_expert`: بهترین Expert برای هر نقطه؛ یک سقف نظری، نه Policy قابل استفاده در Runtime

## نکته پژوهشی مهم

بهتر بودن MoE تضمین ریاضی ندارد. اگر MoE از بهترین Expert بهتر نشود، باید موارد زیر بررسی شوند:

- Complementarity متخصص‌ها
- کیفیت Prior
- Budget
- تعداد Episodeها
- پوشش Calibration
- حساسیت ضرایب Gating

این نسخه عمداً نتیجه را دست‌کاری نمی‌کند؛ بلکه چارچوبی فراهم می‌کند که بتوان دقیقاً علت موفقیت یا شکست MoE را تحلیل کرد.
