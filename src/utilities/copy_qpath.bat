@echo off
set "source_directory=D:\03 Cancer Bioinformatics\LN-project\data"
set "destination_directory=C:\Users\hooll\Dropbox\19 PostGrad\Genomic Medicine\Research Module\Data\annotations\solo"
set "folders_list=59 60 58 62 66 69 72 76 78 77 80 81 84 86 90 93 100 103 104"

for %%F in (%folders_list%) do (
  xcopy /s /i "%source_directory%\%%F" "%destination_directory%\%%F"
)
