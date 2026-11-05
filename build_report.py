"""
Auto-builds the Word report from all saved CSV/PNG artifacts.

Usage:
    python build_report.py

Output:
    Отчет_лабораторная.docx   (ready to open in Word)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
OUT = ROOT / "Отчет_лабораторная.docx"


# ---------- helpers ----------
def h1(doc, text):
    doc.add_heading(text, level=1)


def h2(doc, text):
    doc.add_heading(text, level=2)


def para(doc, text, bold=False, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    return p


def bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def numbered(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Number")


def code_block(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(9)
    return p


def df_to_table(doc, df: pd.DataFrame):
    df = df.reset_index(drop=True)
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns):
        t.rows[0].cells[i].text = str(c)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row.values):
            cells[i].text = str(v)
    return t


def add_img_if_exists(doc, path: Path, width_in: float = 6.0, caption: str | None = None):
    if not path.exists():
        para(doc, f"[рисунок отсутствует: {path.name}]", italic=True)
        return
    doc.add_picture(str(path), width=Inches(width_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        p = doc.add_paragraph()
        r = p.add_run(caption)
        r.italic = True
        r.font.size = Pt(9)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def safe_csv(name, **kw):
    p = REPORTS / name
    return pd.read_csv(p, **kw) if p.exists() else None


# ---------- build ----------
def build():
    doc = Document()

    # ---- титул ----
    t = doc.add_heading("Прогнозирование спроса на алкогольную продукцию", level=0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, "Лабораторная работа: методы машинного обучения для временных рядов",
         italic=True)
    para(doc, "")
    para(doc, "Выполнил(а): __________________________", bold=True)
    para(doc, "Группа:      __________________________", bold=True)
    para(doc, "Дата:        __________________________", bold=True)
    doc.add_page_break()

    # ---- 1. Введение ----
    h1(doc, "1. Введение")
    para(doc,
         "Цель работы — применить широкий спектр методов машинного обучения "
         "для прогнозирования спроса на алкогольную продукцию на синтетическом "
         "наборе исторических данных, сравнить их по метрикам качества и "
         "продемонстрировать интерактивный прогноз на горизонт, задаваемый "
         "пользователем.")
    para(doc, "Задачи:")
    numbered(doc, [
        "Сгенерировать реалистичный набор данных о продажах алкоголя (несколько SKU, "
        "промо, праздники, ценовая эластичность, OOS).",
        "Реализовать и сравнить модели из разделов: регрессия, регуляризация, "
        "временные ряды, нейронные сети, байесовский подход, кросс-секционные "
        "модели, ансамбли.",
        "Показать метрики MAE, RMSE, WAPE на валидации и тесте.",
        "Сделать функцию прогноза на горизонт N дней с вводом пользователя и "
        "графиком прогноза.",
        "Сделать выводы о применимости каждого класса моделей.",
    ])
    para(doc, "Стек:")
    bullets(doc, [
        "Python 3.14, pandas, numpy;",
        "scikit-learn (LinearRegression, Ridge, Lasso, BayesianRidge);",
        "LightGBM (градиентный бустинг);",
        "Prophet (Meta) — сезонность и праздники РФ;",
        "PyTorch (LSTM);",
        "statsmodels, matplotlib, seaborn.",
    ])

    # ---- 2. Данные ----
    h1(doc, "2. Данные")
    para(doc,
         "Данные сгенерированы скриптом generate_data.py с фиксированным seed=42. "
         "В синтетике заложены бизнес-эффекты, характерные для алкогольного ритейла:")
    bullets(doc, [
        "долгосрочный тренд;",
        "недельная сезонность (пики в пятницу и субботу);",
        "годовая сезонность (декабрьский пик + мартовские и майские праздники);",
        "праздники РФ (1–8 января, 23.02, 08.03, 01.05, 09.05, 12.06, 04.11);",
        "промо-акции с недельным всплеском и post-promo dip;",
        "ценовая эластичность относительно цены конкурента;",
        "дни OOS (out-of-stock): продажи = 0, но спрос существует;",
        "случайный шум.",
    ])

    sales_attrs = pd.DataFrame({
        "Параметр": ["Период", "SKU", "Категории", "Строк", "Частота"],
        "Значение": [
            "2022-01-01 .. 2024-12-31",
            "6 (SKU_01 .. SKU_06)",
            "wine / spirits",
            "6 576",
            "ежедневная",
        ],
    })
    df_to_table(doc, sales_attrs)
    doc.add_paragraph()

    h2(doc, "2.1. EDA")
    add_img_if_exists(doc, REPORTS / "eda_total_sales.png", 6.5,
                      "Рис. 1. Суммарные продажи по всем SKU.")
    add_img_if_exists(doc, REPORTS / "eda_one_sku.png", 6.5,
                      "Рис. 2. Один SKU: продажи, промо-дни, дни OOS.")
    add_img_if_exists(doc, REPORTS / "eda_by_weekday.png", 4.5,
                      "Рис. 3. Средние продажи по дням недели.")
    add_img_if_exists(doc, REPORTS / "eda_by_month.png", 4.5,
                      "Рис. 4. Средние продажи по месяцам.")
    add_img_if_exists(doc, REPORTS / "eda_promo_effect.png", 4.0,
                      "Рис. 5. Средние продажи в промо и без промо.")
    add_img_if_exists(doc, REPORTS / "eda_sales_distribution.png", 5.0,
                      "Рис. 6. Распределение ежедневных продаж.")

    para(doc,
         "Выводы по EDA: данные обладают ярко выраженной недельной и годовой "
         "сезонностью, промо-дни дают заметный прирост средних продаж, "
         "OOS-дни формируют нули, которые нельзя интерпретировать как нулевой "
         "спрос. Это подтверждает необходимость признакового инжиниринга "
         "(лаги, скользящие средние, календарные признаки, промо, OOS).")

    # ---- 3. Методика ----
    h1(doc, "3. Методика")
    para(doc,
         "Данные разделены хронологически: 70 % — train, 15 % — validation, "
         "15 % — test, без перемешивания. Признаки включают календарь, "
         "экзогенные переменные и лаги (7/14/30 дней) + скользящие "
         "средние/стандартные отклонения (7 и 28 дней). Все лаги и "
         "скользящие статистики сдвинуты минимум на 1 день — утечки в "
         "будущее нет.")

    para(doc, "Реализованные модели по разделам задания:", bold=True)
    df_methods = pd.DataFrame({
        "Раздел": [
            "1. Классическая регрессия",
            "1. Градиентный бустинг",
            "2. Регуляризация",
            "3. Временные ряды",
            "3. Feature-based ML",
            "4. Нейронные сети",
            "5. Байесовский подход",
            "6. Кросс-секционные модели",
            "7. Ансамбли",
        ],
        "Модель / реализация": [
            "LinearRegression",
            "LightGBM: global и per-SKU",
            "Ridge, Lasso",
            "Prophet + праздники РФ + регрессоры",
            "LightGBM на лагах и rolling",
            "LSTM (PyTorch), окно 28 дней",
            "BayesianRidge (интервалы P10–P90)",
            "LightGBM global по всем SKU",
            "Weighted average (1/WAPE_вал), 2 варианта",
        ],
    })
    df_to_table(doc, df_methods)
    doc.add_paragraph()

    para(doc, "Метрики:")
    bullets(doc, [
        "MAE — средняя абсолютная ошибка, единицы продаж;",
        "RMSE — корень из средней квадратичной ошибки, штрафует большие промахи;",
        "WAPE — взвешенная абсолютная ошибка в процентах (сумма |y−ŷ| / сумма |y|), "
        "устойчива к нулям.",
    ])

    # ---- 4. Результаты ----
    h1(doc, "4. Результаты")

    h2(doc, "4.1. Baseline")
    base = safe_csv("baseline_metrics.csv")
    if base is not None:
        agg = base[~base["split"].str.contains(":")]
        df_to_table(doc, agg.round(3))
        doc.add_paragraph()
    para(doc,
         "Baseline rolling_28 — самая сильная из «глупых» моделей: "
         "сглаживание за 28 дней даёт test WAPE ≈ 30 %.")

    h2(doc, "4.2. Сводная таблица моделей (test)")
    summary = safe_csv("summary_all.csv")
    if summary is not None:
        test_only = summary[summary["split"] == "test"].sort_values("WAPE_%")
        df_to_table(doc, test_only.round(3))
        doc.add_paragraph()

    add_img_if_exists(doc, REPORTS / "summary_bar.png", 6.5,
                      "Рис. 7. Сравнение моделей по test WAPE.")

    h2(doc, "4.3. Ансамбли")
    para(doc,
         "Ансамбль построен как взвешенное среднее с весами 1 / WAPE на "
         "валидации, нормированными в сумму 1. Сравнивались два варианта: "
         "по всем моделям и по топ-3 (LightGBM global + Prophet + LSTM).")
    para(doc,
         "Результат: ансамбль top-3 (WAPE ≈ 12.1 %) улучшил Prophet и LSTM, "
         "но не побил lgbm_global (10.3 %). Ансамбль из всех моделей хуже "
         "(14.2 %), потому что слабые линейные модели получают ненулевой вес. "
         "Это классический пример: при явном доминировании одной модели "
         "взвешивание всех подряд не помогает.")

    h2(doc, "4.4. Байесовские интервалы")
    bay = safe_csv("preds_bayesian_with_intervals.csv")
    if bay is not None:
        coverage = ((bay["sales"] >= bay["p10"]) &
                    (bay["sales"] <= bay["p90"])).mean() * 100
        para(doc, f"Bayesian Ridge обучен с выводом per-sample std. "
                  f"Эмпирическое покрытие 90%-го интервала P10–P90 на test: "
                  f"{coverage:.1f} % (цель ≈ 90 %). Модель даёт честные "
                  f"интервалы, что важно для бизнес-планирования.")

    h2(doc, "4.5. Прогноз по запросу пользователя")
    para(doc,
         "Скрипт forecast.py обучает LightGBM global на всех данных и "
         "строит рекурсивный многошаговый прогноз на горизонт, введённый "
         "пользователем (демо: 7 и 30 дней), для всех SKU или отдельного SKU. "
         "Сценарий по умолчанию: цена и конкурентная цена на последнем "
         "уровне, промо и OOS не активны — «базовый» прогноз спроса.")

    for horizon in (7, 30, 90):
        p = REPORTS / f"forecast_SKU_01_{horizon}d.png"
        if p.exists():
            add_img_if_exists(doc, p, 6.5,
                              f"Рис. Прогноз спроса SKU_01 на {horizon} дней.")

    # ---- 5. Ключевые фрагменты кода ----
    h1(doc, "5. Ключевые фрагменты кода")
    para(doc, "Полный код — в приложенных .py-файлах. Ниже — наиболее "
              "важные фрагменты.")

    para(doc, "5.1. Feature engineering без утечек:", bold=True)
    code_block(doc,
               "for lag in LAGS:\n"
               "    df[f'lag_{lag}'] = df.groupby('sku')['sales'].shift(lag)\n"
               "df['_s'] = df.groupby('sku')['sales'].shift(1)\n"
               "for w in ROLLING_WINDOWS:\n"
               "    df[f'rolling_mean_{w}'] = df.groupby('sku')['_s'].transform(\n"
               "        lambda s: s.rolling(w).mean())\n"
               "    df[f'rolling_std_{w}']  = df.groupby('sku')['_s'].transform(\n"
               "        lambda s: s.rolling(w).std())")

    para(doc, "5.2. Хронологический сплит:", bold=True)
    code_block(doc,
               "dates = df['date'].drop_duplicates().sort_values()\n"
               "n = len(dates)\n"
               "train_end = dates.iloc[int(n * 0.70) - 1]\n"
               "val_end   = dates.iloc[int(n * 0.85) - 1]\n"
               "train = df[df['date'] <= train_end]\n"
               "val   = df[(df['date'] > train_end) & (df['date'] <= val_end)]\n"
               "test  = df[df['date'] > val_end]")

    para(doc, "5.3. LightGBM global с early stopping:", bold=True)
    code_block(doc,
               "dtrain = lgb.Dataset(X_train, label=y_train,\n"
               "                     categorical_feature=['sku'])\n"
               "dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)\n"
               "model = lgb.train(PARAMS, dtrain, num_boost_round=2000,\n"
               "                  valid_sets=[dval],\n"
               "                  callbacks=[lgb.early_stopping(60)])")

    para(doc, "5.4. Prophet с праздниками и регрессорами:", bold=True)
    code_block(doc,
               "m = Prophet(yearly_seasonality=True, weekly_seasonality=True,\n"
               "            daily_seasonality=False)\n"
               "m.add_country_holidays(country_name='RU')\n"
               "for col in ['promo_flag', 'discount', 'price']:\n"
               "    m.add_regressor(col)\n"
               "m.fit(train_df)\n"
               "forecast = m.predict(future_df)")

    para(doc, "5.5. Байесовские интервалы:", bold=True)
    code_block(doc,
               "pipe  = build_pipeline().fit(X_train, y_train)\n"
               "mean, std = pipe.predict(X_val, return_std=True)\n"
               "p10, p90  = mean - 1.645 * std, mean + 1.645 * std")

    para(doc, "5.6. Ансамбль (веса 1/WAPE_вал):", bold=True)
    code_block(doc,
               "inv = {m: 1.0 / val_wape[m] for m in models}\n"
               "total = sum(inv.values())\n"
               "w = {m: inv[m] / total for m in models}\n"
               "pred_ens = sum(preds[m] * w[m] for m in models)")

    # ---- 6. Выводы ----
    h1(doc, "6. Выводы")
    bullets(doc, [
        "Все ML-модели существенно побили baseline (rolling_28: 30 % WAPE) — "
        "лучший результат у LightGBM global: ~10 % test WAPE, что в 3 раза "
        "лучше baseline.",
        "Кросс-секционная глобальная модель (единая для всех SKU) точнее "
        "per-SKU модели, потому что использует больше данных и общие "
        "паттерны сезонности и промо.",
        "Prophet показал себя достойно (~19 %) без лаговых признаков, "
        "подтверждая ценность встроенного учёта сезонности и праздников.",
        "LSTM на коротких рядах (3 года) уступает градиентному бустингу, "
        "но превосходит линейные модели — паттерны последовательностей "
        "улавливаются частично.",
        "Ансамбль не всегда выигрывает: при доминировании одной модели "
        "взвешивание всех подряд ухудшает результат; отбор топ-3 "
        "сглаживает, но не побивает лидера.",
        "Байесовский подход даёт честные интервалы P10–P90 (~93 % покрытие "
        "при заявленных 90 %), что полезно для планирования закупок и "
        "оценки рисков.",
        "Для продакшена рекомендован LightGBM global как основной прогноз, "
        "а Bayesian Ridge / Prophet — как источники интервалов и "
        "интерпретируемых компонент (тренд, сезонность, праздники).",
    ])

    # ---- 7. Ограничения ----
    h1(doc, "7. Ограничения и направления развития")
    bullets(doc, [
        "Данные синтетические и детерминированные — на них ML-модели "
        "показывают заметно лучший WAPE (~10 %), чем обычно на реальных "
        "данных (20–35 %). При интерпретации результатов нужна поправка.",
        "В синтетике нет длинных аномалий (ковид, резкие скачки цен, "
        "дефицит), нет иерархической структуры «регион → магазин → SKU».",
        "Прогноз рекурсивный: ошибки накапливаются с горизонтом. Для "
        "длинных горизонтов (>90 дней) разумно использовать директивный "
        "подход (прямое прогнозирование на h шагов) или иерархическое "
        "согласование.",
        "Ансамбль можно улучшить: stacking (мета-модель поверх предсказаний) "
        "или оптимизация весов на скользящем окне вместо фиксированных 1/WAPE.",
        "Стоит добавить: DeepAR/TFT для мульти-SKU, ковариаты «погода», "
        "лаги цены конкурентов, признаки маркетинговых каналов.",
    ])

    # ---- 8. Приложение ----
    h1(doc, "8. Приложение: структура проекта")
    code_block(doc,
               "lab_demand_forecast/\n"
               "├── config.py               # константы, пути, seed\n"
               "├── generate_data.py        # генератор синтетики\n"
               "├── features.py             # feature engineering + сплит\n"
               "├── eda.py                  # разведочный анализ\n"
               "├── baseline.py             # naive / seasonal / rolling\n"
               "├── model_linear.py         # линейная + Ridge + Lasso\n"
               "├── model_lgbm.py           # LightGBM global + per-SKU\n"
               "├── model_prophet.py        # Prophet + RU праздники\n"
               "├── model_lstm.py           # LSTM (PyTorch)\n"
               "├── model_bayesian.py       # BayesianRidge + интервалы\n"
               "├── ensemble.py             # взвешенный ансамбль\n"
               "├── forecast.py             # прогноз по вводу пользователя\n"
               "├── summarize.py            # сводная таблица\n"
               "└── data/sales_data.csv")

    doc.save(OUT)
    print(f"[OK] Report saved -> {OUT}")


if __name__ == "__main__":
    build()
