#ifndef AESCRYPTO_H
#define AESCRYPTO_H

#include <iostream>
#include <QtCore/QString>
#include <QtCore/QByteArray>
#include <QtCore/QCryptographicHash>
#include <QtCore/QFile>
#include <openssl/aes.h>
#include <glog/logging.h>

#if defined USE_CAFFE
#include <caffe/caffe.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#endif

#define AES_BITS 128

class AESCrypto
{
public:
    AESCrypto();
    ~AESCrypto();
    std::stringstream aesCryptoStringStream(int mode, std::stringstream &modelCache, QString key);
    int aesCryptoFile(int mode, QString file, QString key);
    bool removeFile(QString file);
    QString code;
#if (defined USE_CAFFE)
    caffe::NetParameter fileDecryptToProto(std::string fileName, QString key);
#endif
private:
    QByteArray getSHA1ofKey(QString key);
};

#endif
