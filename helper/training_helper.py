import helper.cifar10_dataset as cifar10_dataset
import helper.cifar100_dataset as cifar100_dataset
import helper.cifar50_dataset as cifar50_dataset
import helper.imagenet_dataset as ImagenetDataset
import helper.mnist_dataset as mnist_dataset
import helper.emnist_dataset as emnist_dataset
import helper.twentynews as twentynews
import helper.DBPedia as DBPedia
import helper.DBPedia_tfidf as DBPedia_tfidf
import helper.aiart_dataset as aiart_dataset

__author__ = 'garrett_local'


def load_dataset(dataset_name):

    if dataset_name == 'mnist':
        return mnist_dataset.MnistDataset
    if dataset_name == 'cifar10':
        return cifar10_dataset.Cifar10Dataset
    if dataset_name == 'cifar100':
        return cifar100_dataset.Cifar100Dataset
    if dataset_name == 'cifar50':
        return cifar50_dataset.Cifar50Dataset
    if dataset_name == 'imagenet':
        return ImagenetDataset.ImagenetIterator
    if dataset_name == 'emnist':
        return emnist_dataset.EMnistDataset
    if dataset_name == 'twentynews':
        return twentynews.twentynews
    if dataset_name == 'dbpedia':
        return DBPedia.DBPedia
    if dataset_name == 'dbpedia_tfidf':
        return DBPedia_tfidf.DBPedia
    if dataset_name == 'aiart':
        return aiart_dataset.AiartDataset
