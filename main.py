from data_loaders.mnist_loaders import get_mnist_loaders
from models.meta import FeedforwardModelGenerator2
import pytorch_lightning as pl

train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=64, num_workers=0)
FeedforwardModelGenerator2.main(epoch_amount=10, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

trainer = pl.Trainer()
trainer.fit(MetaModel, train_loader, val_loader)

