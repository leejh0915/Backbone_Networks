# from core.backbone.SimpleNet import SimpleNet
# from data.fashion_mnist import Fashion_mnist
# import matplotlib.pyplot as plt
#
# if __name__ == "__main__":
#     # load fashion_mnist data
#     fashion_mnist = Fashion_mnist()
#     model = SimpleNet()
#     x_train, y_train, x_test, y_test = fashion_mnist.get_data()
#
#     # compile and train
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#
#
#     history = model.fit(x_train, y_train, epochs=10, validation_split=0.25)
#
#     # model evaluate
#     test_loss, test_acc = model.evaluate(x_test, y_test)
#
#     print(test_acc)

from core.backbone.ResNet.ResNet import ResNet50
from data.tensorflow.fashion_mnist import Fashion_mnist

if __name__ == "__main__":
    #load fashion_mnist data
    fashion_mnist = Fashion_mnist()
    model = ResNet50(num_classes=10)
    x_train, y_train, x_test, y_test = fashion_mnist.get_data()

    # compile and train
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


    history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.25)

    # model evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print(test_acc)

