from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import matplotlib.pylab as plt
plt.switch_backend('TkAgg')

def main():
    model_name = 'oriented_obb_ellipse'
    parser = ArgumentParser()
    parser.add_argument('--img', default='/home/hnu1/ZDS/ellipse/test/images/003.jpg')
    parser.add_argument('--config', default='./configs/FAIR1M/' + model_name + '.py')
    parser.add_argument('--checkpoint', default='./work_dir/' + model_name + '/epoch_12.pth')
    parser.add_argument(
        '--device', default='cuda:1', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr, show=True)


if __name__ == '__main__':
    main()
