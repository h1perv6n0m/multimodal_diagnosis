import os
import pandas as pd
import numpy as np

def load_data(filepath: str):
    """
    :param filepath: путь к xlsx-файлу с рамановскими спектрами
    :return: кортеж (metadata, spectra, wavelengths)
    """
    df = pd.read_excel(filepath, decimal=',')

    meta_cols = ['id', 'group', 'class', 'class1', 'type', 'type1']
    metadata = df[meta_cols].copy()

    spectra = df.drop(columns=meta_cols).copy()

    wavelengths = []
    for col in spectra.columns:
        try:
            wavelengths.append(float(str(col).replace(',', '.')))
        except ValueError:
            wavelengths.append(np.nan)
    wavelengths = np.array(wavelengths)

    spectra = spectra.apply(pd.to_numeric, errors='coerce')

    return metadata, spectra, wavelengths

def print_report(metadata, spectra, wavelengths):
    """
    :param metadata: таблица с метаданными образцов
    :param spectra: матрица спектральных данных
    :param wavelengths: массив длин волн
    :return: None
    """
    print("=" * 60)
    print("Отчёт по датасету")
    print("=" * 60)

    print(f"\nВсего строк (образцов): {len(metadata)}")
    print(f"Уникальных пациентов (id): {metadata['id'].nunique()}")

    print("\nРаспределение по class1:")
    print(metadata['class1'].value_counts().to_string())

    print("\nРаспределение по type1:")
    print(metadata['type1'].value_counts().to_string())

    counts = metadata['id'].value_counts()
    multi = counts[counts > 1]
    print(f"\nПациентов с несколькими образцами: {len(multi)}")
    print(f"Всего образцов у таких пациентов: {multi.sum()}")

    print(f"\nДиапазон длин волн: от {np.nanmin(wavelengths):.2f} до {np.nanmax(wavelengths):.2f}")
    print(f"Спектральных точек: {spectra.shape[1]}")

    print(f"\nПропусков в метаданных: {metadata.isna().sum().sum()}")
    print(f"Пропусков в спектрах: {spectra.isna().sum().sum()}")

    bad_waves = np.isnan(wavelengths).sum()
    if bad_waves > 0:
        print(f"\nВНИМАНИЕ: {bad_waves} столбцов с нечисловыми длинами волн")

    print("=" * 60)

def save_processed(metadata, spectra, out_dir: str):
    """
    :param metadata: таблица с метаданными
    :param spectra: матрица спектральных данных
    :param out_dir: папка для сохранения
    :return: None
    """
    os.makedirs(out_dir, exist_ok=True)
    metadata.to_csv(os.path.join(out_dir, 'metadata.csv'), index=False)
    spectra.to_csv(os.path.join(out_dir, 'spectra.csv'), index=False)
    print(f"\nСохранено в {out_dir}/metadata.csv и {out_dir}/spectra.csv")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    filepath = os.path.join(project_root, 'data', 'spectra', 'onco_Raman_spectra.xlsx')
    out_dir = os.path.join(project_root, 'data', 'processed')

    metadata, spectra, wavelengths = load_data(filepath)
    print_report(metadata, spectra, wavelengths)
    save_processed(metadata, spectra, out_dir)