clear all; clc;

generation_done = 1;

path_to_pcd = "/home/yerzhan/Desktop/zmp/kaibo/kaibo_recordeddata/data/MPC_Todai/recdata_2018-08-22T10-25-53/recdata_20180822_012558_003375/pcl_rs16Rec_20180822_0129/";
files = dir(sprintf("%s%s", path_to_pcd, "*.pcd"));

files = {files.name};
files = sort(files);
fileAmount = numel(files);

pathFolders = path_to_pcd(strfind(path_to_pcd, "MPC_Todai") + 10:end - 1);
pathFoldersList = strsplit (pathFolders, "/");

foldName = "";
for i=1:numel(pathFoldersList)
  foldName = strcat(foldName, pathFoldersList{i});
  foldName = strcat(foldName, "/");
  system(strcat("mkdir /home/yerzhan/temp/", foldName));
end

outDir = strcat("/home/yerzhan/temp/", pathFolders);
outDir = strcat(outDir, "/");

if (!generation_done)
  for i=1:fileAmount
    myfile = files(i){1};
    csvFileName = myfile(1:end - 4);
    csvFileName = strcat(csvFileName, ".csv");
    
    system(sprintf("/home/yerzhan/projects/pcd2csv/pcd2csv %s%s %s%s", path_to_pcd, myfile, outDir, csvFileName));
  end
end

%% part 2

files = dir(sprintf("%s%s", outDir, "*.csv"));
files = {files.name};
files = sort(files);
fileAmount = numel(files);

res = 10;

out_dir = strcat(outDir, "grid_files/");

for i=1:fileAmount
  myfile = files(i){1};
  velo = importdata(strcat(outDir, myfile), ' ');
  
  % remove all points behind image plane (approximation
  idx = velo(:,1)<5;
  velo(idx,:) = [];
    
  % convert pcl to grid file with the specified resolution
  grid = pcltogrid(velo, res);
    
  filename = sprintf('%s%06d.csv', out_dir, i);
  csvwrite(filename, grid);
end
