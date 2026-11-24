from src.data.streams import load_data


def load_cifar_c(configuration, corruption, severity, transform):
    if configuration["type"] != "cifar_c" or not 1 <= severity <= 5:
        raise ValueError("Expected a CIFAR-C dataset and severity in [1,5]")
    return load_data(configuration, corruption, severity, transform)[0]
