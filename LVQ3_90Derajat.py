import math
import pandas as pd
import random
def normalisasiPengujian(data):
    Max = []
    Min = []
    for i in range(len(data)):
        Max.append(max(data[i]))
        Min.append(min(data[i]))
    listBaru = []
    for i in range(len(data)):
        x = []
        for j in range(len(data[i])):
            normal = (data[i][j]-Min[i]) / (Max[i]-Min[i])
            x.append(normal)
        listBaru.append(x)
    return(listBaru)

def duplicateAndShuffle(datasets, multiplier):
    ##duplicate
    temp=[]
    for i in range(multiplier):
        for data in datasets:
            temp+=[data]
    if(temp==[]):
        temp=datasets
    random.shuffle(temp)
    return temp

def normalisasiTraining(data):
    max=[-1,-1,-1,-1,-1,-1]
    min=[-1,-1,-1,-1,-1,-1]
    for i in range(len(data)):
        for j in range(6):
            if(max[j]==-1 and min[j]==-1):
                max[j]=data[i][j]
                min[j] = data[i][j]
            else:
                if (max[j]<data[i][j]):
                    max[j] = data[i][j]
                if(min[j]>data[i][j]):
                    min[j] = data[i][j]
    for i in range(len(data)):
        for j in range(6):
            data[i][j]=(data[i][j]-min[j])/(max[j]-min[j])
    return data

def euclideanDistance(x,y):
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
    return(distance)

def cekTarget(listJ,D):
    if listJ.index(D) + 1 == 1:
        nilai = 2
    elif listJ.index(D) + 1 == 2:
        nilai = 1
    else:
        nilai = 3
    return(nilai)

def training(baru,kelas,bobotAwal,window,learningRate,minLearningRate):
    Epoch = 1
    for epoch in range(1000):
        for i in range(20):
            hitungJ = []
            for j in range(6):
                hitungJ.append(baru[i][j])
            J1 = euclideanDistance(hitungJ, bobotAwal[0])
            J2 = euclideanDistance(hitungJ, bobotAwal[1])
            J3 = euclideanDistance(hitungJ, bobotAwal[2])
            listJ = [J1, J2, J3]
            copylistJ = [J1, J2, J3]
            D1 = min(copylistJ)
            copylistJ.remove(D1)
            D2 = min(copylistJ)
            T1 = cekTarget(listJ, D1)
            T2 = cekTarget(listJ, D2)
            if T1 == kelas[i]:
                for k in range(len(hitungJ)):
                    bobotAwal[listJ.index(D1)][k] += learningRate * (hitungJ[k] - bobotAwal[listJ.index(D1)][k])
            elif (kelas[i] == T2):
                if ((min(D1 / D2, D2 / D1)) > ((1 - window) * (1 + window))):
                    for k in range(len(hitungJ)):
                        bobotAwal[listJ.index(D1)][k] -= learningRate * (hitungJ[k] - bobotAwal[listJ.index(D1)][k])
                        bobotAwal[listJ.index(D2)][k] += learningRate * (hitungJ[k] - bobotAwal[listJ.index(D2)][k])
                else:
                    for k in range(len(hitungJ)):
                        bobotAwal[listJ.index(D1)][k] -= (window * learningRate) * (
                                    hitungJ[k] - bobotAwal[listJ.index(D1)][k])
                        bobotAwal[listJ.index(D2)][k] += (window * learningRate) * (
                                    hitungJ[k] - bobotAwal[listJ.index(D2)][k])
            else:
                for k in range(len(hitungJ)):
                    bobotAwal[listJ.index(D1)][k] -= learningRate * (hitungJ[k] - bobotAwal[listJ.index(D1)][k])
        learningRate = learningRate * 0.1

        if learningRate < minLearningRate:
            Epoch = epoch
            break
        Epoch = epoch
    print("Total Epoch\t\t\t:", Epoch)
    print("Bobot Akhir\t\t\t:", bobotAwal)
    return bobotAwal

def pengujian(data,kelas,bobot):
    kelasUjiLatih = []
    akurasi = 0
    for i in range(len(data[0])):
        hitungJ2 = []
        for j in range(6):
            hitungJ2.append(data[j][i])
        UJ1 = euclideanDistance(hitungJ2, bobot[0])
        UJ2 = euclideanDistance(hitungJ2, bobot[1])
        UJ3 = euclideanDistance(hitungJ2, bobot[2])
        listJ2 = [UJ1, UJ2, UJ3]
        if min(listJ2) == UJ1:
            kelasUjiLatih.append(2)
        elif min(listJ2) == UJ2:
            kelasUjiLatih.append(1)
        else:
            kelasUjiLatih.append(3)
    print("Ekspektasi Target Kelas\t\t\t:", kelas)
    print("Hasil Target Kelas\t\t\t\t:", kelasUjiLatih)
    for i in range(len(kelas)):
        if kelasUjiLatih[i] == kelas[i]:
            akurasi += (1 / len(kelas))
    print("Akurasi Pengujian Data Latih\t:", akurasi * 100)

def RUN():
    data_train=pd.read_csv("Data Latih.csv")
    newData=[]
    for data in range(len(data_train["target"])):
        temp=[]
        for features in data_train:
            temp+=[data_train[features][data]]
        newData+=[temp]
    data_train=newData
    dataLatih=data_train
    baru = normalisasiTraining(dataLatih)
    baru= duplicateAndShuffle(baru,10)
    kelas=[]
    for i in range(len(baru)):
        kelas+=[baru[i][6]]
    bobotAwal = [[0.9,0.2,0.1,0.3,0.9,0.2],[0.7,0.5,0.2,0.1,0.8,0.4],[0.5,0.9,0.5,0.7,0.7,0.6]]
    learningRate = 0.1
    minLearningRate = 0.000000000000000001
    window = 0.35
    bobotAkhir = training(baru,kelas,bobotAwal,window,learningRate,minLearningRate)
    dataUjiLatih = [
                 [1116855072,909402453,1017703456,972132664,1048334448,101899777,953204013,1037973244,1051491639,918240803,878956522,813149387,949405797,975843924,561575251,974537347,654005574,697975474,755811594,627283166],
                 [0.0539,0.0606,0.0597,0.0594,0.0589,0.0576,0.0587,0.0586,0.0572,0.0643,0.0648,0.0683,0.0617,0.0627,0.0735,0.0568,0.0673,0.0654,0.0617,0.0702],
                 [0.3593,0.3886,0.3640,0.3776,0.3642,0.3692,0.3811,0.3710,0.3699,0.3838,0.3921,0.3994,0.3758,0.3759,0.4661,0.3850,0.4222,0.4324,0.4124,0.4481],
                 [0.9440,0.9470,0.9293,0.9384,0.9336,0.9380,0.9428,0.9386,0.9413,0.9328,0.9375,0.9336,0.9342,0.9303,0.9641,0.9577,0.9512,0.9560,0.9553,0.9566],
                 [209176143,189570792,201849498,202362319,202241918,200483835,191670011,208316611,21374359,19661427,189154961,18519175,197694537,199523969,14154961,211956522,166123746,164023411,166023411,155114827],
                 [0.0033,0.0041,0.0040,0.0039,0.0039,0.0038,0.0039,0.0037,0.0036,0.0046,0.0046,0.0050,0.0043,0.0044,0.0061,0.0035,0.0050,0.0047,0.0044,0.0055]
                 ]
    dataNormal = normalisasiPengujian(dataUjiLatih)
    ekspektasiUji = [2,2,2,2,2,2,2,1,1,1,1,1,1,1,3,3,3,3,3,3]
    print('===========================================================================')
    print("PENGUJIAN DATA UJI")
    pengujian(dataNormal,ekspektasiUji,bobotAkhir)
    dataUji = [
       [1068180602,997424749,1032424749,993405797,875852843,815685619,702244147,681536232,686833891],
       [0.0552,0.0600,0.0570,0.0582,0.0674,0.0678,0.0637,0.0638,0.0642],
       [0.3657,0.3707,0.3692,0.3715,0.3904,0.4033,0.4203,0.4260,0.4279],
       [0.9424,0.9352,0.9395,0.9412,0.9289,0.9354,0.9558,0.9590,0.9589],
       [201117057,199555184,205075808,209392419,195053512,181975474,172733556,168369008,170128205],
       [0.0035,0.0040,0.0036,0.0037,0.0050,0.0051,0.0045,0.0045,0.0045]
       ]
    ekspektasiUji = [2,2,2,1,1,1,3,3,3]
    dataNormal = normalisasiPengujian(dataUji)
    print('===========================================================================')
    print("PENGUJIAN DATA UJI")
    pengujian(dataNormal,ekspektasiUji,bobotAkhir)

if __name__ == '__main__':
    RUN()