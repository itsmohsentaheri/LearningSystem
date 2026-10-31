# معلم تطبیقی مبتنی بر AI — Adaptive Curriculum Tutor

یک سیستم سه‌مرحله‌ای برای تبدیل سیلابس به دورهٔ یادگیری تطبیقی:

1. **سازندهٔ گراف دروس** (`curriculum_graph_builder.py`) — سیلابس متنی را با کمک LLM (OpenRouter) به یک گراف وابستگی DAG از درس‌ها تبدیل می‌کند و در `curriculum_graph.json` ذخیره می‌کند.
2. **معلم زمان‌اجرا** (`runtime_adaptive_tutor.py`) — روی همان JSON، جلسهٔ تطبیقی برگزار می‌کند: تشخیص سطح سناریومحور، میکرودرس، آزمون‌های متنوع (MCQ / cloze / تشریحی کوتاه)، جبران خطا (remediation)، آزمون فصلی، و **تکرار فاصله‌دار FSRS**.
3. **رابط گرافیکی مرورگر** (`gui_server.py`) — همان منطق CLI را بدون تغییر در یک وب‌اپ Flask تعاملی (RTL فارسی) اجرا می‌کند.

```
test.txt ──build──▶ curriculum_graph.json ──run──▶ learner_profile*.json
                          ▲                            │
                     gui_server.py ◀── تعامل مرورگر ───┘
```

## نصب

```bash
python -m venv .venv && source .venv/bin/activate   # اختیاری
pip install -r requirements.txt                       # networkx, requests, pydantic, flask
```

تنظیم کلید API برای حالت لایو (یا داخل فایل `.env` که خودکار بارگذاری می‌شود):

```bash
export OPENROUTER_API_KEY="sk-or-..."
export OPENROUTER_MODEL="openai/gpt-4o-mini"          # اختیاری
```

بدون کلید، همهٔ ابزارها با `--simulate` کاملاً آفلاین و قابل‌اجرا هستند (محتوای mock قطعی).

## ۱) ساخت گراف سیلابس

```bash
python curriculum_graph_builder.py --syllabus test.txt            # لایو (نیازمند کلید API)
python curriculum_graph_builder.py --simulate                     # نمونهٔ آفلاین
```

خروجی: `curriculum_graph.json` شامل گره‌های درس، پیش‌نیازها (DAG)، هدف هر ماژول و مرتب‌سازی توپولوژیک. چرخه‌ها خودکار ترمیم و یال‌های زائد حذف می‌شوند.

## ۲) اجرای معلم در ترمینال (CLI)

```bash
python runtime_adaptive_tutor.py --graph curriculum_graph.json
python runtime_adaptive_tutor.py --simulate                 # تست آفلاین end-to-end
python runtime_adaptive_tutor.py --user-id sara             # پروفایل جدا: learner_profile_sara.json
python runtime_adaptive_tutor.py --check-due-reviews        # مرورهای FSRS معوق (JSON) + توضیح XAI
python runtime_adaptive_tutor.py --update-portfolio         # ویرایش پروفایل شخصی‌سازی
```

### چه چیزی در طول جلسه اتفاق می‌افتد؟
- **فرم پروفایل (portfolio):** نام، هدف، پیشینه، ترجیحات → برای شخصی‌سازی عمق درس‌ها.
- **تشخیص سطح:** سؤال‌های سناریومحور چندموضوعی؛ فقط «عمق» درس را تنظیم می‌کند، هیچ سرفصلی حذف نمی‌شود.
- **میکرودرس + آزمون:** نوع سؤال بر اساس شواهد یادگیرنده انتخاب می‌شود (streak موفق ← تشریحی، خطا ← cloze). پاسخ غلط → بازخورد بدفهمی + تلاش دوباره/توضیح.
- **آزمون فصلی:** پس از تکمیل هر فصل؛ نمرهٔ پایین آن فصل را «weak» علامت می‌زند.
- **FSRS (فعال):** هر_attempt_ آزمون و هر مرور، وضعیت حافظهٔ موضوع (stability/difficulty/فاصله) را به‌روز می‌کند؛ مرورهای معوق **قبل از درس‌های جدید** در ابتدای جلسهٔ بعد ارائه می‌شوند و فراموشی مجدد → درس مروری فوری.
- **Resume:** هر خروج = ذخیرهٔ اتمی پروفایل؛ اجرای بعدی دقیقاً ادامه می‌دهد.

## ۳) رابط گرافیکی (GUI)

```bash
python gui_server.py                      # http://127.0.0.1:8000
python gui_server.py --simulate           # محتوای آفلاین، تعامل انسانی در مرورگر
python gui_server.py --port 9000 --graph curriculum_graph.json
```

صفحه را باز کنید و **«شروع / ادامه جلسه»** را بزنید. GUI یک آداپتر I/O به‌جای کنسول به `tutor.run()` تزریق می‌کند؛ یعنی نمره‌دهی، FSRS، جبران خطا و ذخیرهٔ پروفایل **دقیقاً همان کد CLI** است.

امکانات صفحه:
- فرم شخصی‌سازی، سؤال‌ها با گزینه‌های کلیکى و ورودی متن آزاد (Enter = ارسال)
- **نوار پیشرفت** + جدول فصل‌ها (تکمیل، نمرهٔ آزمون فصل، هشدار فصل ضعیف)
- **جدول مرورهای FSRS**: زمان مرور بعدی، فاصلهٔ روزها، تعداد مرور، علامت «معوق»
- **لاگ زنده** و رفرش خودکار وضعیت هر ۸ ثانیه
- پشتیبانی چند کاربری: هنگام شروع، شناسهٔ کاربر وارد کنید → `learner_profile_<id>.json` جدا

### API های داخلی (برای توسعه)
| Route | نقش |
|---|---|
| `POST /api/start` | ساخت جلسه (user_id / simulate) و اجرای thread معلم |
| `POST /api/next?timeout=25` | long-poll رویداد بعدی: `form / question / action / info / done / error / heartbeat` |
| `POST /api/answer` | ارسال پاسخ (`kind`: text/choice/action/form) |
| `GET /api/status` | نمای پیشرفت: فصل‌ها، FSRS، portfolio، تاریخچه |

سرور تک‌جلسه‌ای است (هر `start` جدید جایگزین قبلی می‌شود) و پیش‌فرض فقط روی `127.0.0.1` bind می‌شود.

## فایل‌ها

| فایل | توضیح |
|---|---|
| `curriculum_graph_builder.py` | سیلابس → گراف درس (Pydantic + NetworkX + OpenRouter) |
| `runtime_adaptive_tutor.py` | موتور معلم تطبیقی + FSRS + CLI |
| `gui_server.py` | سرور Flask و UI مرورگر (RTL) |
| `test.txt` | سیلابس نمونه (ابزارهای AI) |
| `curriculum_graph.json` | گراف تولیدشده (خروجی مرحلهٔ ۱) |
| `learner_profile*.json` | وضعیت هر یادگیرنده (خروجی مرحلهٔ ۲/۳) |
| `requirements.txt` | وابستگی‌ها |

## عیب‌یابی سریع
- **«OpenRouter API key not found»** → کلید را export کنید یا `--simulate` اضافه کنید.
- **«No active session»** در مرورگر → ابتدا «شروع / ادامه جلسه».
- پروفایل خراب/منقضی → فایل `learner_profile*.json` مربوطه را پاک کنید تا جلسه از صفر شروع شود (یا `--user-id` جدید بدهید).
- GUI روی پورت اشغال‌شده بالا نمی‌آید → `--port 8080`.

## محدودیت‌های شناخته‌شده
- سرور GUI برای استفادهٔ محلی/کلاس کوچک طراحی شده (توسعهٔ Flask، تک‌جلسه‌ای، بدون احراز هویت).
- نمره‌دهٔ تشریحی کوتاه، تطبیق نرمال‌شدهٔ رشته است (نه semantic)؛ پاسخ‌های مترادفِ غیرثبت‌شده ممکن است رد شوند و مسیر self-rating برای همین وجود دارد.
- تست خودکار (pytest) هنوز نوشته نشده؛ حداقل مسیر قابل‌اتکا `--simulate` دو ابزار است.
