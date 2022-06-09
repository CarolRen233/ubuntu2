function index=subroutine(pathNCCTImage,PathNCCT_Brain)
disp(['Strip skull of patient ' pathNCCTImage]);

% load the subject image
ImgSubj_nii = load_untouch_nii(pathNCCTImage);
ImgSubj_hdr = ImgSubj_nii.hdr;
ImgSubj = ImgSubj_nii.img;
%ImgSubj = double(ImgSubj);

% skull stripping
NCCT_Thr = 100; % for NCCT images
CTA_Thr = 400; % for CTA images

[brain] = SkullStripping(double(ImgSubj),NCCT_Thr);

% save image
Output_nii.hdr = ImgSubj_hdr;
Output_nii.img = int16(brain);
for x=1:size(brain,1)
    if(max(max(brain(x,:,:)))>0)
        x_min=x;
        break;
    end
end
for y=1:size(brain,2)
    if(max(max(brain(:,y,:)))>0)
        y_min=y;
        break;
    end
end
for z=1:size(brain,3)
    if(max(max(brain(:,:,z)))>0)
        z_min=z;
        break;
    end
end
for x=size(brain,1):-1:1
    if(max(max(brain(x,:,:)))>0)
        x_max=x;
        break;
    end
end
for y=size(brain,2):-1:1
    if(max(max(brain(:,y,:)))>0)
        y_max=y;
        break;
    end
end
for z=size(brain,3):-1:1
    if(max(max(brain(:,:,z)))>0)
        z_max=z;
        break;
    end
end
index=[x_min,y_min,z_min,x_max,y_max,z_max];
save_nii(Output_nii, PathNCCT_Brain);

disp([pathNCCTImage '----skull tripping finished']);