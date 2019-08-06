#include <aescrypto.h>

using namespace std;

AESCrypto::AESCrypto()
{
    code = "ropeok123";
}
AESCrypto::~AESCrypto()
{}

QByteArray AESCrypto::getSHA1ofKey(QString key)
{
    QString pwd=key;
    QByteArray ba,bb;
    QCryptographicHash md(QCryptographicHash::Sha1);
    ba.append(pwd);
    md.addData(ba);
    bb = md.result();
    return bb;
}

stringstream AESCrypto::aesCryptoStringStream(int mode, stringstream &modelCache, QString key)
{
    QByteArray SHA1Key = QByteArray(getSHA1ofKey(key));

    unsigned char aeskey[16];
    for (int i = 0; i < 16; i++) {
        aeskey[i] = SHA1Key.at(i);
    }

    AES_KEY aes;
    stringstream errorReturn;
    errorReturn << "error";
    if (mode == AES_ENCRYPT) {
        if(AES_set_encrypt_key((unsigned char*)aeskey, 128, &aes) < 0) {
            LOG(ERROR) << "Set aes key for encrypting model failed";
            return errorReturn;
        }
    } else if (mode == AES_DECRYPT) {
        if(AES_set_decrypt_key((unsigned char*)aeskey, 128, &aes) < 0) {
            LOG(ERROR) << "Set aes key for decrypting model failed";
            return errorReturn;
        }
    } else {
         LOG(ERROR) << "Crypto mode input error";
        return errorReturn;
    }

    modelCache.seekg(0, ios::end);
    int modelSize = modelCache.tellg();
    modelCache.seekg(0, ios::beg);

    unsigned char* modelMem = (unsigned char*)malloc(modelSize);
    if (modelMem == nullptr) {
        LOG(ERROR) << "Alloc memory for crypting model failed.";
        return errorReturn;
    }

    modelCache.read((char*)modelMem, modelSize);

    unsigned char iv[AES_BLOCK_SIZE];
    for (int j = 0; j < AES_BLOCK_SIZE; j++) {
        iv[j]=0;
    }
    int len=16;
    long pos = 0;

    for (pos = pos; pos + 16 <= modelSize; pos += 16) {
        AES_cbc_encrypt((unsigned char*)&modelMem[pos], (unsigned char*)&modelMem[pos], len, &aes, iv, mode);
    }
    stringstream output;
    output.seekg(0, ios::beg);
    output.write((char*)modelMem, modelSize);
    free(modelMem);

    if (mode == AES_ENCRYPT) {
        LOG(INFO) << "Model encrypted";
    } else {
        LOG(INFO) << "Model decrypted";
    }
    return output;
}

int AESCrypto::aesCryptoFile(int mode, QString file, QString key)
{
    QByteArray SHA1Key = QByteArray(getSHA1ofKey(key));

    unsigned char aeskey[16];
    for (int i = 0; i < 16; i++) {
        aeskey[i] = SHA1Key.at(i);
    }

    AES_KEY aes;

    if (mode == AES_ENCRYPT) {
        if(AES_set_encrypt_key((unsigned char*)aeskey, 128, &aes) < 0) {
            LOG(ERROR) << "Set aes key for encrypting model file failed";
            return -1;
        }
    } else if (mode == AES_DECRYPT) {
        if(AES_set_decrypt_key((unsigned char*)aeskey, 128, &aes) < 0) {
            LOG(ERROR) << "Set aes key for decrypting model file failed";
            return -1;
        }
    } else {
         LOG(ERROR) << "Crypto mode input error";
        return -1;
    }

    QFile inputFile(file);
    if (!inputFile.open(QIODevice::ReadOnly)) {
        LOG(ERROR) << "Open model file " << file.toStdString() <<" failed";
        return -1;
    }
    QByteArray fileByteArray = QByteArray(inputFile.readAll());
    inputFile.close();
    QByteArray outputArray;
    outputArray.resize(fileByteArray.size());
    long fileSize = fileByteArray.size();

    unsigned char iv[AES_BLOCK_SIZE];
    for(int j = 0; j < AES_BLOCK_SIZE; j++) {
        iv[j]=0;
    }
    int len=16;
    unsigned char in[16];
    unsigned char out[16];
    long pos = 0;

    for (pos = pos; pos + 16 <= fileSize; pos += 16) {
        for (int j = 0; j < 16; j++) {
            in[j] = fileByteArray.at(pos + j);
        }
        AES_cbc_encrypt((unsigned char*)in, (unsigned char*)out, len, &aes, iv, mode);
        for (int j = 0; j < 16; j++) {
            outputArray.data()[pos + j] = out[j];
        }
    }
    for(int i = pos; i < fileSize; i++) {
        outputArray.data()[i] = fileByteArray.at(i);
    }

    QString endName;
    if (mode == AES_ENCRYPT) {
        endName = QString(".enc");
    }
    if (mode == AES_DECRYPT) {
        endName = QString(".dec");
    }
    QFile writeFile(file + endName);
    if (!writeFile.open(QIODevice::WriteOnly)) {
        LOG(ERROR) << "Open write model file " << (file + endName).toStdString() << " failed";
        return -1;
    }
    writeFile.write(outputArray);
    writeFile.flush();
    writeFile.close();

    return 0;
}

bool AESCrypto::removeFile(QString file)
{
    if(QFile::remove(file)) {
        return true;
    }
        
    LOG(ERROR) << "Remove file " << file.toStdString() << " failed";
    return false;
}

#if (defined USE_CAFFE)
caffe::NetParameter AESCrypto::fileDecryptToProto(std::string fileName, QString key)
{
    caffe::NetParameter proto;
    LOG(INFO) << "Read model file " << fileName << " to decrypt";
    ifstream fin;
    fin.open(fileName);
    string fileContent;
    if (fin.is_open()) {
        stringstream temp;
        temp<<fin.rdbuf();
        temp = aesCryptoStringStream(AES_DECRYPT, temp, key);
        fileContent= temp.str();
        fin.close();
    }
    istringstream netpare(fileContent);
    google::protobuf::io::IstreamInputStream * net_input = new google::protobuf::io::IstreamInputStream((std::istream *)(&netpare));
    google::protobuf::io::CodedInputStream* coded_input_p = new google::protobuf::io::CodedInputStream(net_input);
    coded_input_p->SetTotalBytesLimit(INT_MAX, 536870912);
    proto.ParseFromCodedStream(coded_input_p);
    return proto;
}
#endif

/*
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    cout << "start processing..." << endl;
    QTime current = QTime::currentTime();

    AESCrypto aes;

    if(QString::compare(argv[1],"-enc") == 0)
        aes.aesCryptoFile(AES_ENCRYPT, argv[2], argv[3]);
    if(QString::compare(argv[1],"-dec") == 0)
        aes.aesCryptoFile(AES_DECRYPT, argv[2], argv[3]);


    QTime now = QTime::currentTime();
    cout << "Elapsed" << current.msecsTo(now) << endl;
    cout << "processed" << endl;
    return a.exec();
}
*/
