from cProfile import run
import os
import sys
import shutil
import argparse
import random
from itertools import combinations

sys.path.insert(0, '../model')
sys.path.insert(0, '../utils')

import numpy as np

import torchvision.transforms as T
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()




import dcgan
import temporal_encoder as TE
import temporal_discriminator as TD

#from dataloader import WoundImageDataset
from dataloader import WoundImagePairsDataset # new dataset with day i and j
from synth_labels import synthesize_softmax_labels as synth_softmax
from synth_labels import synthesize_onehot_labels as synth_onehot



parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches") # changed from 16 to 8
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate") # changed from 0.0002
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space") # original: 100, new: 16
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")  # changed from 64 to 128
parser.add_argument('--n_classes', type=int, default=16, help='number of classes for dataset')
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=250, help="interval betwen image samples")
opt = parser.parse_args()


IMG_SHAPE = (opt.channels, opt.img_size, opt.img_size)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP = 0.02
D_UPDATE_THRESHOLD = 2.0



def _extrapolate_linear(x1, x2, idx_i, idx_j, idx_k):
    diff = x2-x1
    dist = (idx_j-idx_i).unsqueeze(1)
    proj_dist = idx_k-idx_i
    unit_diff = diff/dist
    proj_diff = unit_diff * proj_dist.unsqueeze(1)
    x3 = x1 + proj_diff
    # print(x1[0, :].to("cpu").numpy())
    # print(x2[0, :].to("cpu").numpy())
    # print(x3[0, :].to("cpu").numpy())
    # print(dist, proj_dist)
    # exit()
    return x3


def list_full_paths(directory, mode="train"):
    return_imgs = []
    return_idxs = []

    def pack_filename(idx, mice):
        original = "Day %d_%s.png"%(idx, mice)
        augmented = "aug_Day %d_%s.png"%(idx, mice)
        return original, augmented
    
    #full_list = [os.path.join(directory, file) for file in os.listdir(directory) if "png" in file]
    full_list = [x for x in os.listdir(directory) if "png" in x]

    # cherry pick test and validation images
    test_imgs = [x for x in full_list if "Y8-4-L" in x or "A8-1-R" in x]
    val_imgs = [x for x in full_list if "Y8-4-R" in x or "A8-1-L" in x]
    train_imgs = set(full_list).difference(set(test_imgs))
    train_imgs = train_imgs.difference(set(val_imgs))
    train_imgs = list(train_imgs)

    if mode=="train": full_list = train_imgs
    elif mode=="val": full_list = val_imgs
    else: full_list = test_imgs

    mouse_ids = set([x.split('_')[-1].split('.')[0] for x in full_list])
    print(mouse_ids)

    for mice in mouse_ids:
        i = 0
        for k in range(15, 1, -1):
            file_k, file_k_aug = pack_filename(k, mice)
            if file_k not in full_list: continue

            j = k
            while j > 1:
                j -= 1
                file_j, file_j_aug = pack_filename(j, mice)
                if file_j not in full_list: continue

                #for i in range(j):
                for i in range(min(8, j)):
                    file_i, file_i_aug = pack_filename(i, mice)

                    if file_i not in full_list: continue

                    path_i, path_j, path_k = os.path.join(directory, file_i), os.path.join(directory, file_j), os.path.join(directory, file_k)
                    path_ia, path_ja, path_ka = os.path.join(directory, file_i_aug), os.path.join(directory, file_j_aug), os.path.join(directory, file_k_aug)

                    A = [x for x in [path_i, path_ia] if os.path.exists(x)]
                    B = [x for x in [path_j, path_ja] if os.path.exists(x)]
                    C = [x for x in [path_k, path_ka] if os.path.exists(x)]
                    grid1 = np.array(A)
                    grid2 = np.array(B)
                    grid3 = np.array(C)

                    grid_3d = np.meshgrid(grid1, grid2, grid3)
                    combs = np.array(grid_3d).T.reshape(-1,3).tolist()
                    return_imgs.extend(combs)
                    for _ in range(len(combs)): return_idxs.append([i, j, k])

    return (return_imgs, return_idxs)





def train_cgan(datapath, annotation_file, outpath="../tmp/"):
    ''' This is the script to train Conditional GAN a.k.a. DCGAN '''
    # Some interesting pages to read while waiting
    #
    # https://www.reddit.com/r/MachineLearning/comments/5asl74/discussion_discriminator_converging_to_0_loss_in/
    # https://ai.stackexchange.com/questions/8885/why-is-the-variational-auto-encoders-output-blurred-while-gans-output-is-crisp
    #
    # These will help you very much

    C = opt.n_classes
    if C not in [4, 16]: raise NotImplementedError("Check n_classes in arguments")
    
    # normalization parameters of Circular Cropped Wound Dataset
    MEAN = torch.tensor([0.56014212, 0.40342121, 0.32133712])
    STD = torch.tensor([0.20345279, 0.14542403, 0.12238597])

    # create output folder
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.mkdir(outpath)

    # retrieve the list of image paths
    #img_list = list_full_paths(datapath)
    train_imgs, train_indices = list_full_paths(datapath, "train")
    #val_imgs = list_full_paths(datapath, "val")
    #test_imgs = list_full_paths(datapath, "test")


    # Loss functions for part 1, 2, and 3
    adversarial_loss = torch.nn.BCELoss()
    #adversarial_loss = torch.nn.MSELoss() # an alternative loss function
    
    if C == 16:
        # if it's 16, they are not binary values and should be treated as random numbers
        embedding_loss = torch.nn.MSELoss() # embedding loss with MSE
        #embedding_loss = torch.nn.CosineEmbeddingLoss() # embedding loss with CEL https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
    else:
        # if it's 4 (stages), then it must be *cross entropy*
        embedding_loss = torch.nn.CrossEntropyLoss()
    
    temporal_loss = torch.nn.BCELoss() #############################################################################
    #temporal_loss = torch.nn.BCEWithLogitsLoss()


    # Initialize Generator and discriminator
    generator = dcgan.Generator(IMG_SHAPE, opt.latent_dim, C)
    discriminator = dcgan.Discriminator(IMG_SHAPE, C)


    # Initialize Temporal Encoder
    temporal_encoder = TE.Classifier_Encoder()
    #temporal_encoder.load_state_dict(torch.load("../checkpoints/normalized_classifier.tar"))
    #temporal_encoder.load_from_state_dict("../checkpoints/normalized_classifier.tar")
    # for p in temporal_encoder.parameters():
    #     p.require_grads = False
    # temporal_encoder.eval()
    temporal_encoder.train()


    # Initialize Temporal Discriminator
    tc_latent_dim = 32
    temporal_discriminator = TD.TemporalDiscriminator(IMG_SHAPE, tc_latent_dim)


    # Initialize weights
    #generator.apply(weights_init_normal)
    #discriminator.apply(weights_init_normal)

    
    print("Using %s.\n"%DEVICE)
    generator = generator.to(DEVICE)
    discriminator = discriminator.to(DEVICE)
    temporal_encoder = temporal_encoder.to(DEVICE)
    temporal_discriminator = temporal_discriminator.to(DEVICE)

    adversarial_loss = adversarial_loss.to(DEVICE)
    embedding_loss = embedding_loss.to(DEVICE)
    temporal_loss = temporal_loss.to(DEVICE)

    MEAN = MEAN.to(DEVICE)
    STD = STD.to(DEVICE)


    # Configure data loaders and compose transform functions
    # https://stackoverflow.com/questions/65676151/how-does-torchvision-transforms-normalize-operates
    TRANSFORMS = T.Compose([T.ToTensor(), \
        T.Resize((opt.img_size, opt.img_size))]) # normalization afterwards

    train_dataset = WoundImagePairsDataset(train_imgs, train_indices, annotation_file, transform = TRANSFORMS)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)


    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


    # densenet image preparation
    DENSENET_IMAGE_SHAPE = 244
    transform_densenet = T.Compose([T.Resize(DENSENET_IMAGE_SHAPE) ])


    batches_done=0
    d_loss, g_loss = 0, 0


    for epoch in range(opt.epochs):
        for i, data in enumerate(train_dataloader):
            
            # PREPARE DATA

            # unload data
            imgs_i, imgs_j, imgs_k = data[0]
            idx_i, idx_j, idx_k = data[1][:, 0].cuda(), data[1][:, 1].cuda(), data[1][:, 2].cuda()
            Y4 = data[2]
            #Y16 = data[3]

            # training parameters
            B = opt.batch_size
            # SKIP BATCH SIZE OF 1
            if imgs_k.shape[0] < B: continue


            # Configure input
            imgs_i = Variable(imgs_i.type(torch.FloatTensor).cuda())
            imgs_j = Variable(imgs_j.type(torch.FloatTensor).cuda())
            imgs_k = Variable(imgs_k.type(torch.FloatTensor).cuda())

            # IMPORTANT: for Discriminators, *standardize* every image so far, both real and generated
            standardize = T.Normalize(MEAN, STD)
            imgs_i      =   standardize(imgs_i)
            imgs_j      =   standardize(imgs_j)
            imgs_k      =   standardize(imgs_k)


            # unfreeze temporal encoder and perform extrapolation on the go
            embeds_i = temporal_encoder(transform_densenet(imgs_i))[1]      # [1] because we only need the embedding from returned (u1, embeddings), where u1 is the class prediction
            embeds_j = temporal_encoder(transform_densenet(imgs_j))[1]
            extp_embeds_k = _extrapolate_linear(embeds_i, embeds_j, idx_i, idx_j, idx_k)
            Y16 = extp_embeds_k

            
            # Sample labels as generator input
            if C == 4:
                real_y = Y4.cuda()
                gen_y = synth_softmax(n_classes=C, batch_size=B).to("cuda")
                #gen_y = synth_onehot(n_classes=C, batch_size=B, sorted=False).to("cuda")
                y_disp = synth_onehot(n_classes=C, batch_size=B, sorted=True).to("cuda") # labels for display
            
            elif C == 16:
                real_y = Y16.cuda()
                gen_y = Y16.cuda()
                y_disp = Y16.cuda() # labels for display
            
            # sample noise for generating fakes
            noise = Variable(torch.randn((B, opt.latent_dim)).cuda())

            # Adversarial ground truths
            valid = Variable(torch.ones(B).cuda(), requires_grad=False)
            fake = Variable(torch.zeros(B).cuda(), requires_grad=False)


            # INITIATE TRAINING

            # =====================================================================================================================
            # Prerequisite: Generate a Batch of Images
            # =====================================================================================================================

            # Generate a batch of images
            gen_imgs = generator(noise, gen_y).view(B, *IMG_SHAPE)

            # IMPORTANT: for Discriminators, *standardize* every image so far, both real and generated
            gen_imgs = standardize(gen_imgs)


            # =====================================================================================================================
            #  Train Discriminator (gen_imgs need to be detached)
            # =====================================================================================================================

            optimizer_D.zero_grad()

            # ----------------------------------   REALISM LOSS (Loss R)

            # Loss for real images
            d_real_loss = adversarial_loss(discriminator(imgs_k, gen_y.detach()).squeeze(), valid) #############################################################################
            # Loss for fake images
            d_fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_y.detach()).squeeze(), fake)
            
            
            # ----------------------------------   TEMPORAL COHERENCE LOSS (Loss Xpred_k | Xi, Xj)
            Yirk = torch.ones((B,1)) # i and real k
            Yjrk = torch.ones((B,1)) # j and real k
            Yifk = torch.zeros((B,1)) # i and fake k
            Yjfk = torch.zeros((B,1)) # j and fake k

            gen_k = gen_imgs.detach()

            """discriminator sees real and fake pairs, generator needs to see the the fake pairs w/ i & j with 'valid' """
            temporal_target = torch.cat((Yirk, Yjrk, Yifk, Yjfk), dim=0).cuda()
            temporal_K = torch.cat((imgs_k, imgs_k, gen_k, gen_k), dim=0)
            temporal_IJ = torch.cat((imgs_i, imgs_j, imgs_i, imgs_j), dim=0)

            temporal_pred = temporal_discriminator(temporal_IJ, temporal_K)
            """ neg of This loss should be added to gen. discriminator sees real and fake pairs"""
            # if epoch > 4:
            #     print(temporal_pred)
            #     print(temporal_target)
            #     exit()
            t_loss = temporal_loss(temporal_pred, temporal_target)

            
            # ----------------------------------   TOTAL DISCRIMINATOR LOSS (L_D)
            
            #g_loss = Lz|c(x)+L(x|x_i, x_j)
            # if epoch > 10 and (d_real_loss < 0.1 or d_fake_loss > 1.0):
            #     d_loss = t_loss
            # else:
            #     
            d_loss = (d_real_loss + d_fake_loss + t_loss) # ALL 3 LOSSES
            #d_loss = d_real_loss + d_fake_loss           # ONLY VANILLA GAN LOSS

            # Finalizing: conditional update D
            # if g_loss <= D_UPDATE_THRESHOLD:   # TEST
            d_loss.backward()
            #torch.nn.utils.clip_grad_norm_(discriminator.parameters(), CLIP) # notice the trailing _ representing in-place
            optimizer_D.step()



            # =====================================================================================================================
            #  Train Generator (DO NOT detach gen_imgs)
            # =====================================================================================================================
            
            # ----------------------------------   EMBEDDING LOSS (Loss yhat|z or c|z)

            #cls_input = gen_imgs.detach()
            cls_input = gen_imgs
            cls_input = transform_densenet(cls_input)

            target = gen_y

            label_pred, fake_embeddings = temporal_encoder(cls_input)         # NOT SURE IF DETACHING AGAIN IS CORRECT FOR LOSS  COMPUTATION; temporal_encoder won't update bc there's no optim for it, but it will accumulate gradients. .detach() prevent gradient accumulation
            
            #prediction = F.softmax(label_pred, dim=-1) if C == 4 else fake_embeddings # label_pred has not been softmaxed yet
            prediction = label_pred if C == 4 else fake_embeddings             # CrossEntropy requires un-normalized data
            
            emb_loss = embedding_loss(target, prediction)
            
            optimizer_G.zero_grad()

            # Loss measures generator's ability to fool the D_realism
            #prediction = discriminator(gen_imgs, gen_y).squeeze()
            """ add emb_loss (not negative) and neg of d_fake_loss and d_real_loss should be added to g_loss. Temporal loss
            needs the needs i&j,k w/ 'real' label
            g_loss = Lz|c(x)-Lr(x)-L(x|x_i, x_j)
            
            # I Don't think you neg_d_real_loss, I think this would just act as a regularizing term
            neg_d_real_loss = adversarial_loss(real, fake)
            neg_d_fake_loss = adversarial_loss(discriminator(gen_imgs, gen_y).squeeze(), real) #already
            g_loss = emb_loss + neg_d_real_loss + neg_d_fake_loss
            """
            # ----------------------------------   FAKE TEMPORAL COHERENCE LOSS (Loss Xpred_k | Xi, Xj)
            gen_k = gen_imgs
            
            g_temporal_target = torch.cat((Yirk, Yjrk), dim=0).cuda() # Real temporal targets
            g_temporal_K = torch.cat((gen_k, gen_k), dim=0) # generated k's
            g_temporal_IJ = torch.cat((imgs_i, imgs_j), dim=0) # real i & j
            
            g_temporal_pred = temporal_discriminator(g_temporal_IJ, g_temporal_K)
            neg_fake_t_loss = temporal_loss(g_temporal_pred, g_temporal_target)

            # ----------------------------------   TOTAL GENERATOR LOSS (L_G)
                            
            neg_d_fake_loss = adversarial_loss(discriminator(gen_imgs, gen_y).squeeze(), valid)
            
            #g_loss = Lz|c(x)-Lr(x)-L(x|x_i, x_j)
            # if epoch > 10 and neg_d_fake_loss < 0.1:
            #     g_loss = emb_loss + neg_fake_t_loss
            # else:
            g_loss = neg_d_fake_loss + emb_loss + neg_fake_t_loss
            #g_loss = emb_loss
            
            g_loss.backward()
            #torch.nn.utils.clip_grad_norm_(generator.parameters(), CLIP) # notice the trailing _ representing in-place
            optimizer_G.step()
            
            # print results
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.epochs, i, len(train_dataloader),
                                                                d_loss.data.cpu(), g_loss.data.cpu()),
                                                                end='\r'
                    )
            # print("emb", emb_loss.item())
            # print("-fake_temporal", neg_fake_t_loss.item())
            # print("(end)")


            # =====================================================================================================================
            # Per Iteration Wrap-up: Present Generated Images and Save Checkpoint
            # =====================================================================================================================

            batches_done = epoch * len(train_dataloader) + i
            if batches_done % opt.sample_interval == 0:
                noise = Variable(torch.randn((B, opt.latent_dim)).cuda())

                # y_disp was defined at the very beginning of training
                gen_imgs = generator(noise, y_disp).view(-1, *IMG_SHAPE)

                save_image(gen_imgs.data, os.path.join(outpath, '%d-%d.png' % (epoch,batches_done)), nrow=8, normalize=False) # nrow = number of img per row, original C, current C//4
                #save_image(imgs_k.data, os.path.join(outpath, '%d-%d.png' % (epoch,batches_done)), nrow=C//2, normalize=False) # real data

                step = batches_done // opt.sample_interval
                writer.add_scalar("d_real_loss", d_real_loss, step)
                writer.add_scalar("d_fake_loss", d_fake_loss, step)
                writer.add_scalar("neg_d_fake_loss", neg_d_fake_loss, step)
                print(d_real_loss.item(), d_fake_loss.item(), t_loss.item(), neg_d_fake_loss.item(), emb_loss.item(), neg_fake_t_loss.item())

        if (epoch+1) % 1 == 0:
            # print(g_temporal_target)
            torch.save(generator, os.path.join(outpath, "cgan_gen.pth"))

    # print(temporal_pred)
    # print(temporal_target)

    #return {"model":[generator, discriminator], "dataloaders":[train_dataloader, val_dataloader, test_dataloader]}
    






if __name__ == "__main__":
    print(opt)
    assert(opt.batch_size > 1)
    
    # configure data and annotation path
    datapath = "../data_augmented/"
    #annotation_file = "../data_augmented/augmented_labels.csv"
    annotation_file = "../data_augmented/augmented_data.csv"

    #l = list_full_paths(datapath, mode="train")
    #print(len(l))

    # train/test the models
    train_cgan(datapath, annotation_file)

    writer.flush()