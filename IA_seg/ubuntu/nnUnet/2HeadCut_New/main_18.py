import os
from copy import deepcopy
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils import *
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from SegNet import VGG16_deconv
from data import Dataset
from torch.utils.data import DataLoader
import cv2
import imageio
from skimage.morphology import disk, dilation,remove_small_holes
import argparse
import shutil
import time



def main():
    args = get_arguments()
    
    datalist = pd.read_csv(args.csv)

    input_2Dbone = os.path.join(args.save_dir, 'testset','input_2Dbone')
    
    brain_mip_2d_save = os.path.join(args.save_dir,  'testset','output_brain_mip_2d')
    just_head_cta_save = os.path.join(args.save_dir, 'cta_img')
    just_head_seg_save = os.path.join(args.save_dir, 'ane_seg')
    just_head_properties_save = './after_headcut_properties_XJ18.pkl'

    maybe_create_path(brain_mip_2d_save)
    maybe_create_path(just_head_cta_save)
    maybe_create_path(just_head_seg_save)
    maybe_create_path(input_2Dbone)
    
    
    ####====== which patients======= 
    
    patients_names = []
    for i in range(len(datalist)):
        patient_name = datalist.iloc[i]['id']
        #if datalist.iloc[i]['subset'] == 'test':
        patients_names.append(patient_name)
    print('patients_names:', len(patients_names),patients_names[:5])
    ####===============================
    #patients_names = ['XJTr0000','XJTr0001','XJTr0002']

    
    

    # load or generate properties and # generate bone_2d_image and update nnunet raw
    #all_properties = get_datalist_properties(patients_names,datalist,args.original_dir,test_dir)
    #save_pickle(all_properties, just_head_properties_save)
    all_properties = load_pickle('F:/codes/ubuntu/nnUnet/file/All_Data_Info_XJ18.pkl')
    
    
    ####===========generate input test 2D images===============
    generate_input_time_spend_txt = []
    for patient in patients_names:
        start1=time.time()
        patient_input_save =os.path.join(input_2Dbone, patient)
        if not os.path.exists(patient_input_save + '_bone_mip2.npz'):
            start1=time.time()
            full_cta = sitk.ReadImage(all_properties[patient]['ori_cta'])
            cta_np = sitk.GetArrayFromImage(full_cta).astype(np.float32)
            hu_channel = np.clip(cta_np, 200, 800)
            bone_win = (hu_channel - 200) / (800 - 200)
            bone_mip0, bone_mip1, bone_mip2 = generate_2Dbone_mask(bone_win)
            end1= time.time()
            interval1 = end1-start1
            generate_input_time_spend_txt.append(patient + ' ' + str(interval1))
            np.savez(patient_input_save + '_bone_mip1.npz', bone_mip1=bone_mip1)
            np.savez(patient_input_save + '_bone_mip2.npz', bone_mip2=bone_mip2)
            #visualize(patient_input_save + '.png',bone_mip0=bone_mip0.numpy(),bone_mip1=bone_mip1,bone_mip2=bone_mip2,)
        
    f1 = open(os.path.join(args.save_dir,'generate_input_time_spend.txt'), 'w')
    for line in generate_input_time_spend_txt:
        f1.write(line + '\n')
    f1.close()
    ####===========================================================
    
    
    
        
    ####===============test loader and model load=====================
    test_dataset = Dataset(patients_names, input_2Dbone, classes=['brain'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16_deconv()
    CLASSES = ['brain']
    model = torch.load('./model/epoch190_model.pth')
    model = model.cuda()
    model.eval()
    print ('test num:',len(test_loader))
    ####==================================================================

    resize_txt = []
    time_spend_txt= []
    for testdata in test_loader:

        shape1, shape2, name, input_image_1,input_image_2 = testdata
        start2 = time.time()
        name = str(name).split("'")[1]
        print('name:', name)

        #visualize(name + '.png', input_image_1 = input_image_1[0][0],input_image_2 = input_image_2[0][0] )

        image_1 = input_image_1.to(device)
        image_2 = input_image_2.to(device)


        with torch.no_grad():
            predict_1 = model(image_1)
            predict_2 = model(image_2)
        

        rows1, cols1 = [int(s) for s in shape1]
        rows2, cols2 = [int(s) for s in shape2]

        print ('rows1, cols1',rows1, cols1)
        print ('rows2, cols2:',rows2, cols2)

        pre_mask_1 = torch.argmax(predict_1.squeeze(), 0)
        pre_mask_2 = torch.argmax(predict_2.squeeze(), 0)

        pre_mask_1 = pre_mask_1 >= 0.5
        output_pre_mask_1 = pre_mask_1.cpu().numpy().astype(int)

        pre_mask_2 = pre_mask_2 >= 0.5
        output_pre_mask_2 = pre_mask_2.cpu().numpy().astype(int)

        #visualize(name + '.png', output_pre_mask_1=output_pre_mask_1, output_pre_mask_2=output_pre_mask_2)

        resized_mask_1 = cv2.resize(output_pre_mask_1, dsize=(cols1, rows1), interpolation=cv2.INTER_NEAREST)
        resized_mask_2 = cv2.resize(output_pre_mask_2, dsize=(cols2, rows2), interpolation=cv2.INTER_NEAREST)
        
        end2 = time.time()
        interval2 = end2-start2

        #visualize(name + '.png', resized_mask_1=resized_mask_1, resized_mask_2=resized_mask_2)

        print ('pre_mask_2 shape:',pre_mask_2.shape)

        one_region_refined_1, largest_removed, kept_size = remove_all_but_the_largest_connected_component(resized_mask_1, [1])
        one_region_refined_2, largest_removed, kept_size = remove_all_but_the_largest_connected_component(resized_mask_2, [1])
        #visualize(os.path.join(brain_mip_2d_save, name + '_befordialation.png'),
        # one_region_refined_1=one_region_refined_1, one_region_refined_2=one_region_refined_2, )

        #visualize(name + '.png', one_region_refined_1=one_region_refined_1, one_region_refined_2=one_region_refined_2)

        #remove_small_holes(one_region_refined_1, area_threshold=8, connectivity=1, in_place=False)
        #remove_small_holes(one_region_refined_2, area_threshold=8, connectivity=1, in_place=False)

        # dilate
        dilate_refined_1 = dilation(one_region_refined_1, selem=disk(20))
        dilate_refined_2 = dilation(one_region_refined_2, selem=disk(20))

        #visualize(name + '.png', dilate_refined_1=dilate_refined_1, dilate_refined_2=dilate_refined_2)

        #visualize(save_f = os.path.join(brain_mip_2d_save, name + '_dilate_refined.png'),
                  #dilate_refined_1=dilate_refined_1, dilate_refined_2=dilate_refined_2,)

        image_1 = cv2.resize(image_1.cpu().numpy().squeeze()[0], dsize=(cols1, rows1), interpolation=cv2.INTER_NEAREST)
        image_2 = cv2.resize(image_2.cpu().numpy().squeeze()[0], dsize=(cols1, rows1), interpolation=cv2.INTER_NEAREST)
        
        interval3 = time.time()-end2
        
        time_spend_txt.append(name + ' ' + str(interval2) + ' ' + str(interval3))

        visualize_overlap2(save_f = os.path.join(brain_mip_2d_save, name + '.png'),mask0=dilate_refined_1, mask1=dilate_refined_2, image_1=image_1,
                          image_2=image_2, )

        print('refined_1.shape:', dilate_refined_1.shape)
        print('refined_2.shape:', dilate_refined_2.shape)

        print ('np.min(np.where(refined_1 > 0)[0]):',np.min(np.where(dilate_refined_1 > 0)[0]))
        print ('np.min(np.where(refined_2 > 0)[0]):',np.min(np.where(dilate_refined_2 > 0)[0]))
        print ('min(np.min(np.where(refined_1 > 0)[0]), np.min(np.where(refined_2 > 0)[0])):',min(np.min(np.where(dilate_refined_1 > 0)[0]), np.min(np.where(dilate_refined_2 > 0)[0])))
        min_brain = min(np.min(np.where(dilate_refined_1 > 0)[0]), np.min(np.where(dilate_refined_2 > 0)[0]))
        max_z = min(np.max(np.where(dilate_refined_1 > 0)[0])+70, np.max(np.where(dilate_refined_2 > 0)[0])+70,dilate_refined_2.shape[0])
        min_z = max(min_brain-70,0)

        min_y = max(np.min(np.where(dilate_refined_2 > 0)[1])-20,0)
        max_y = min(np.max(np.where(dilate_refined_2 > 0)[1])+20,dilate_refined_2.shape[1])
        min_x = max(np.min(np.where(dilate_refined_1 > 0)[1])-20,0)
        max_x = min(np.max(np.where(dilate_refined_1 > 0)[1])+20,dilate_refined_1.shape[1])

        #image_1,image_1,image_2 = image_1.numpy().squeeze(), image_1.numpy().squeeze(), image_2.numpy().squeeze()
        #min_z, max_z, min_y, max_y, min_x, max_x = get_brain_coord(image_1,image_1,image_2 ,brain_max_z, brain_min_z)
        cta_nii = sitk.ReadImage(all_properties[name]['ori_cta'])
        cta_np = sitk.GetArrayFromImage(cta_nii).astype(np.float32)
        seg_nii = sitk.ReadImage(all_properties[name]['ori_seg'])
        seg_np = sitk.GetArrayFromImage(seg_nii).astype(np.int32)

        brain_cta = get_cube_as_coords(cta_np,min_z, max_z, min_y, max_y, min_x, max_x)
        brain_seg = get_cube_as_coords(seg_np, min_z, max_z, min_y, max_y, min_x, max_x)

        print ('before shape:',cta_np.shape)
        print ('after shape:',brain_cta.shape)

        #brain_cta_nii = save_itk_from_numpy(brain_cta, cta_nii)
        #brain_seg_nii = save_itk_from_numpy(brain_seg, seg_nii)

        brain_cta_nii = save_itk_from_numpy(brain_cta, all_properties[name])
        brain_seg_nii = save_itk_from_numpy(brain_seg, all_properties[name])

        resize_txt.append(name + ' ' + str(min_z) + ' '+ str(max_z) +
                          ' '+ str(min_y) + ' '+ str(max_y) +
                          ' '+ str(min_x) + ' '+ str(max_x))
        
        
        all_properties[name]['coords'] = min_z, max_z, min_y, max_y, min_x, max_x
        all_properties[name]['before_size'] = cta_np.shape
        all_properties[name]['after_size'] = brain_cta.shape
        all_properties[name]['just_head_cta_save'] = os.path.join(just_head_cta_save,name+'.nii.gz')
        all_properties[name]['just_head_seg_save'] = os.path.join(just_head_seg_save, name + '.nii.gz')

        sitk.WriteImage(brain_cta_nii, all_properties[name]['just_head_cta_save'])
        sitk.WriteImage(brain_seg_nii, all_properties[name]['just_head_seg_save'])

        visualize(os.path.join(just_head_cta_save,name+'.png'),x=brain_cta[:,:,brain_cta.shape[2]//2],
        z=brain_cta[brain_cta.shape[0]//2,:,:],y=brain_cta[:,brain_cta.shape[1]//2,:])

    save_pickle(all_properties, just_head_properties_save)
    shutil.copyfile('./after_headcut_properties_XJ18.pkl',os.path.join(args.save_dir, 'after_headcut_properties_XJ18.pkl'))
    f1 = open('resize.txt', 'w')
    for line in resize_txt:
        f1.write(line + '\n')
    f1.close()
    
    f2 = open(os.path.join(args.save_dir,'XJ18_time_spend.txt'), 'w')
    for line in time_spend_txt:
        f2.write(line + '\n')
    f2.close()
    
    
    # update nnunet raw
    '''
    if args.update_nn:
        for i in range(len(datalist)):
            patient_name = datalist.iloc[i]['id']
            if patient_name in patients_names:
                data_itk = all_properties[patient_name]['ori_cta']
                nnunet_raw_update(args.nnunet_raw,name,datalist.iloc[i]['subset'],data_itk)
     '''
        
   
    

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--original_dir', type=str, required=False, default='F:/data/xianjin_data',
                        help='the folder include cta_img and ane_seg')
    parser.add_argument('-csv', '--csv', type=str, required=False, default='./XJ18_All_Data_Info.csv',
                        help='ids')
    parser.add_argument('-o', '--save_dir', type=str, required=False, default='F:/codes/ubuntu/nnUnet/2HeadCut_New/XJ18_headcut',
                        help='output folder')
    #parser.add_argument('-nn', '--nnunet_raw', type=str, required=False, default= '/root/workspace/renyan/output/nnUNet/nnUNet_raw_data/Task154_CellPress110new_justhead',
                        #help='output folder')
    
    #parser.add_argument('--update_nn', action='store_true', default=False, help='update nnunet raw data')


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()






