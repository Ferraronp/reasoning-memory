# Проверка поставки v0.2.1

В среде подготовки выполнены:

- 20 unittest-проверок: удаление тела из следующего входа при сохранении архива;
  сохранение полного контекста; независимость парных состояний; восстановление и
  повторная свёртка; отключённое/неизвестное/повторное восстановление; дубли ID;
  запрет финала до эксперимента; некорректная/вложенная разметка; контекстный и
  событийный лимиты; неполный вывод; отсутствие эталона во входе модели.
- Регрессия пользовательского EOS-сбоя: один shared failure, без фиктивных веток
  и их accuracy; корректная агрегация старых unpairable-логов. Guided-фазы,
  общий summary, удаление тела перед финальным продолжением, запрет свёртки
  обрезанного summary и EOS без границы эксперимента.
- CLI `run --mode compact` и `pair` на mock-backend: завершены, ответ 5,
  журналы и агрегаты записаны.
- Разбор всех .py и кодовых ячеек ноутбука через Python AST; разбор pyproject TOML.

В среде подготовки не выполнены: установка HF-зависимостей, реальная модельная генерация,
GPU/FP16/INT8 проверки, открытие в Colab, измерения качества гипотезы.
В среде подготовки нет PyTorch/Transformers. Следующий обязательный технический
шаг — реальный smoke в GPU-сессии. Текущий код HF backend опирается на документацию,
а не на успешно проведённый здесь интеграционный запуск.

По присланным пользователем логам предыдущей версии: реальная Qwen3-0.6B FP16
загрузилась и сгенерировала 241 токен до EOS, но не соблюла автономный протокол.
Новый guided-режим на реальных весах ещё не проверен.

В v0.2.1 дополнительно проверены отсутствие XML-префикса в guided-входе и честный отказ на пустом эксперименте. Новый prompt на реальных весах здесь не проверен.

## Two-stage regression coverage
Scripted generation verifies that followup is absent from stage 1; forks share
summary and continuation seed; stage 2 compact input excludes e1 body; final
compact input excludes both bodies while the archive retains both. A truncated
second summary preserves its unfinished experiment and cannot yield an answer.
Missing followup and followup used with incompatible protocols are rejected.
