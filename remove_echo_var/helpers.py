import numpy as np
import pandas as pd
def read_excel_data_zero_echo(path):

    data = pd.read_excel(path)

    unique_values = set()

    # 'eko bulguları' sütunundaki benzersiz değerleri topla
    # (mapping'i AYNEN hesaplıyoruz ki vektör boyutu 55 olarak korunsun)
    for i in list(data['eko bulguları']):
        if type(i) == float: continue
        for j in i.split(','):
            unique_values.add(j.strip())

    mapping = {v: idx for idx, v in enumerate(list(unique_values))}

    data['NT proBNP'] = (data['NT proBNP'] - data['NT proBNP'].mean()) / data['NT proBNP'].std()

    train_val_test_features = {"train": {}, "val": {}, "test": {}}

    for _, row in data.iterrows():
        # --- ABLATION: eko bulguları multi-hot bloğu bilerek SIFIR bırakılıyor ---
        # idx hesaplama ve record[idx] = 1 satırı kaldırıldı.
        # Vektör boyutu len(mapping) + 4 = 55 olarak korunuyor.
        record = np.zeros(len(mapping) + 4)

        # İlk len(mapping) boyut (eko bulguları) sıfır kalır.
        # Son 4 sayısal özellik aynen korunur (H/CL dahil):
        record[-4:] = [row['kreatinin'], row['NT proBNP'] if row['NT proBNP'] != 'yok' else 0, row['eko EF'], row['3. SAAT K/KL']]

        train_val_test_features[row['Split']][row['HASTA ADI']] = record.tolist()

    return train_val_test_features