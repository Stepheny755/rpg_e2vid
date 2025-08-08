python run_reconstruction.py \
  -c pretrained/E2VID_lightweight.pth.tar \
  -i /mnt/d/Datasets/harmeda/harmeda_dataset/pattern/pattern_ev.hdf5 \
--auto_hdr --output /home/steph/datasets/harmeda --display --fixed_duration


python run_reconstruction.py \
  -c pretrained/E2VID_lightweight.pth.tar \
  -i /mnt/d/Datasets/harmeda/harmeda_dataset/logo/logo_baseline.hdf5 \
--auto_hdr --output /home/steph/datasets/harmeda/logo/ev --display --fixed_duration --no-recurrent


python run_reconstruction.py \
  -c pretrained/E2VID_lightweight.pth.tar \
  -i /mnt/d/Datasets/harmeda/harmeda_dataset/logo/logo_harmeda.hdf5 \
--auto_hdr --output /home/steph/datasets/harmeda/logo/harmeda --display --fixed_duration --no-recurrent
