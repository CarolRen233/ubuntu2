addpath('NIfTI_20140122');
outputFileName=['F:\data\Medzoo_code_use_data\IACTA\brain_coords3.txt'];
logFid=fopen('F:\data\Medzoo_code_use_data\IACTA\log3.txt','a');
outputFid=fopen(outputFileName,'a');
file=dir('F:\data\Medzoo_code_use_data\IACTA\Brain\nii\*.nii');
index=zeros(length(file),6);
for n=1:length(file)
    outputFile=dir('F:\data\Medzoo_code_use_data\IACTA\Brain\*.nii');
    flag=0;
    for i=1:length(outputFile)
        if(strcmp(file(n).name(1:end-4),outputFile(i).name(1:end-10)))
            flag=1;
            continue;
        end
    end
    if(flag==1)
        continue;
    end
    try
        pathNCCTImage = ['F:\data\Medzoo_code_use_data\IACTA\Brain\nii\',file(n).name];
        PathNCCT_Brain = ['F:\data\Medzoo_code_use_data\IACTA\Brain\',file(n).name(1:end-7),'_brain.nii'];
        index(n,:)=subroutine(pathNCCTImage,PathNCCT_Brain);
        fprintf(outputFid,[file(n).name,'\t']);
        dlmwrite(outputFileName,index(n,:),'-append','delimiter','\t','newline','pc');
    catch
        fprintf(logFid,[file(n).name,'\n']);
    end
end
fclose(logFid);
fclose(outputFid);


