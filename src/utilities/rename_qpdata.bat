@echo off

set "list=59:U_90112_15_C_LOW_10_L1 60:U_90112_18_F_LOW_13_L1 58:U_90112_7_X_LOW_4_L2 62:U_90157_19_F_NA_12_L1 66:U_90264_8_F_LOW_8_L1 69:U_90360_10_E_LOW_10_L1 72:U_90424_7_X_MID_4_L2 76:U_90603_14_B_LOW_14_L1 78:U_90653_18_F_NA_18_L1 77:U_90653_6_X_LOW_6_L1 80:U_90670_19_F_NA_11_L1 81:U_100029_8_X_MID_8_L1 84:U_100042_12_X_LOW_7_L2 86:U_100165_4_X_LOW_4_L1 90:U_100195_3_X_LOW_3_L1 93:U_100203_4_X_MID_4_L1 100:U_100237_28_F_LOW_15_L1 103:U_100246_10_B_LOW_10_L1 104:U_100254_3_X_LOW_3_L1"
set "merge_directory=C:\Users\hooll\Dropbox\19 PostGrad\Genomic Medicine\Research Module\Data\annotations\solo"


for %%t in (%list%) do (
    REM Split each tuple into folder_name and new_file_name
    for /F "tokens=1,2 delims=:" %%a in ("%%t") do (
        echo Folder name: %%a
        echo New file name: %%b
        REM Add your logic here for renaming or any other operations

		REM recursively searches for the file data.qpdata within the current folder.
		echo copying "%merge_directory%\%%a\data.qpdata" to "%merge_directory%\%%b.qpdata"
		copy "%merge_directory%\%%a\data.qpdata" "%merge_directory%\%%b.qpdata"
		

    )
)

