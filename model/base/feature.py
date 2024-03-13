r""" Extracts intermediate features from given backbone network & layer ids """


def extract_feat_vgg(img, backbone, feat_ids, bottleneck_ids=None, lids=None):
    r""" Extract intermediate features from VGG """
    feats = []
    feat = img
    for lid, module in enumerate(backbone.features):
        feat = module(feat)
        if lid in feat_ids:
            feats.append(feat.clone())
    return feats


def extract_feat_res(img, backbone, feat_ids, bottleneck_ids, lids):  # img torch.Size([8, 3, 400, 400])
    # feat_ids 要提取特征的层次编号列表；
    # bottleneck_ids：ResNet中每个层次（layer）的瓶颈块（bottleneck block）编号列表
    # ids：ResNet中每个层次（layer）的编号列表
    feats = []

    # Layer 0
    feat = backbone.conv1.forward(img)  # 对图像进行卷积操作   torch.Size([8, 64, 200, 200])
    feat = backbone.bn1.forward(feat)   # 对卷积结果进行批归一化操作  torch.Size([8, 64, 200, 200])
    feat = backbone.relu.forward(feat)  # 激活     torch.Size([8, 64, 200, 200])
    feat = backbone.maxpool.forward(feat) # 最大池化   torch.Size([8, 64, 100, 100])

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)   # torch.Size([8, 256, 100, 100])

        if bid == 0:
            res = backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

        feat += res

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

    return feats
