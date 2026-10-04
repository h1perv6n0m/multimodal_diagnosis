import os
import pandas as pd
import numpy as np


def load_processed(data_dir: str):
    """
    :param data_dir: путь к папке с обработанными данными
    :return: кортеж (metadata, spectra)
    """
    metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    spectra = pd.read_csv(os.path.join(data_dir, 'spectra.csv'))
    return metadata, spectra


def get_wavelengths(spectra):
    """
    :param spectra: матрица спектров с заголовками-длинами волн
    :return: numpy-массив длин волн
    """
    wavelengths = []
    for col in spectra.columns:
        try:
            wavelengths.append(float(str(col).replace(',', '.')))
        except ValueError:
            wavelengths.append(np.nan)
    return np.array(wavelengths)


def build_report(metadata, spectra, wavelengths) -> str:
    """
    :param metadata: таблица с метаданными
    :param spectra: матрица спектров
    :param wavelengths: массив длин волн
    :return: строка с текстовым отчётом
    """
    lines = []
    lines.append('=' * 60)
    lines.append('Описание датасета')
    lines.append('=' * 60)
    lines.append('')

    n_samples = len(metadata)
    n_patients = metadata['id'].nunique()
    counts = metadata['id'].value_counts()
    n_multi = (counts > 1).sum()

    lines.append('Общая информация')
    lines.append(f'Всего образцов: {n_samples}')
    lines.append(f'Уникальных пациентов: {n_patients}')
    lines.append(f'Пациентов с несколькими образцами: {n_multi}')
    lines.append('')

    lines.append('Распределение по классам (class1)')
    class_counts = metadata['class1'].value_counts()
    for cls, cnt in class_counts.items():
        share = cnt / n_samples * 100
        lines.append(f'  {cls}: {cnt} ({share:.1f}%)')
    lines.append('')

    lines.append('Распределение по группам (group)')
    group_counts = metadata['group'].value_counts()
    for grp, cnt in group_counts.items():
        share = cnt / n_samples * 100
        lines.append(f'  {grp}: {cnt} ({share:.1f}%)')
    lines.append('')

    lines.append('Спектральные характеристики')
    lines.append(f'Количество спектральных отсчётов: {spectra.shape[1]}')
    lines.append(f'Диапазон длин волн: от {np.nanmin(wavelengths):.2f} до {np.nanmax(wavelengths):.2f} см^-1')

    diffs = np.diff(wavelengths[~np.isnan(wavelengths)])
    if len(diffs) > 0:
        step_mean = diffs.mean()
        step_std = diffs.std()
        lines.append(f'Средний шаг между точками: {step_mean:.2f} см^-1 (СКО: {step_std:.4f})')
        if step_std < 0.01:
            lines.append('Шаг равномерный')
        else:
            lines.append('Шаг неравномерный')
    lines.append('')

    lines.append('Пропуски в метаданных')
    na_counts = metadata.isna().sum()
    na_counts = na_counts[na_counts > 0]
    if len(na_counts) == 0:
        lines.append('  Пропусков нет')
    else:
        for col, cnt in na_counts.items():
            lines.append(f'  {col}: {cnt}')
    lines.append('')

    lines.append('Пропуски в спектрах')
    na_spectra = spectra.isna().sum().sum()
    lines.append(f'  Всего пропусков: {na_spectra}')
    lines.append('')

    lines.append('=' * 60)
    return '\n'.join(lines)


def save_report(report: str, out_dir: str):
    """
    :param report: текст отчёта
    :param out_dir: папка для сохранения
    :return: None
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'dataset_description.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'Отчёт сохранён: {path}')


def save_class_table(metadata, out_dir: str):
    """
    :param metadata: таблица с метаданными
    :param out_dir: папка для сохранения
    :return: None
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(metadata)

    class_counts = metadata['class1'].value_counts().reset_index()
    class_counts.columns = ['class1', 'count']
    class_counts['share_%'] = (class_counts['count'] / n * 100).round(2)
    path = os.path.join(out_dir, 'class_distribution.csv')
    class_counts.to_csv(path, index=False)
    print(f'Таблица классов сохранена: {path}')

    group_counts = metadata['group'].value_counts().reset_index()
    group_counts.columns = ['group', 'count']
    group_counts['share_%'] = (group_counts['count'] / n * 100).round(2)
    path = os.path.join(out_dir, 'group_distribution.csv')
    group_counts.to_csv(path, index=False)
    print(f'Таблица групп сохранена: {path}')


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    data_dir = os.path.join(project_root, 'data', 'processed')
    out_dir = os.path.join(project_root, 'experiments', 'metrics')

    metadata, spectra = load_processed(data_dir)
    wavelengths = get_wavelengths(spectra)

    report = build_report(metadata, spectra, wavelengths)
    print(report)

    save_report(report, out_dir)
    save_class_table(metadata, out_dir)