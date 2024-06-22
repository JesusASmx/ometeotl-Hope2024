<div align="center">

  # ometeotl@Hope2024

</div>


<div align="center">
  <img src="ometeotl.png">
</div>


In this repository you can find the code to reproduce the submission of Team Ometeotl at Hope2024@IberLEF2024

<a href="https://codalab.lisn.upsaclay.fr/competitions/17714"> Click here to visit the competition's website </a>

## Achievements:

<table align="center">
  <tr>
    <th>Dataset</th>
    <th>F1-score</th>
    <th>Place</th>
    <th>How many from the first place</th>
  </tr>
    
  <tr>
    <td>HopeEDI</td>
    <td>0.64</td>
    <td>4th</td>
    <td>0.03</td>
  </tr>

  <tr>
    <td>PolyHope (Es)</td>
    <td>0.81</td>
    <td>6th</td>
    <td>0.04</td>
  </tr>

  <tr>
    <td>PolyHope (En)</td>
    <td>0.82</td>
    <td>8th</td>
    <td>0.06</td>
  </tr>
</table>


## Reproducibility instructions

<ol>
  <li>
    Clone this repo:
    
```
git clone https://github.com/JesusASmx/ometeotl-Hope2024 YourFolder
```
  </li>
  
  <li>

Put the .csv files of the datasets into the folder ```.YourFolder/Dataset``` with these names:
    <ul>
      <li>```PolyES_train.csv``` for the training split of the PolyHope dataset in spanish.</li>
      <li>```PolyES_train.csv``` for the test split of the PolyHope dataset in spanish.</li>
      <li>```PolyEN_train.csv``` for the training split of the PolyHope dataset in english.</li>
      <li>```PolyEN_train.csv``` for the test split of the PolyHope dataset in english.</li>
      <li>```EDI_train.csv``` for the training split of the HopeEDI dataset.</li>
      <li>```EDI_train.csv``` for the test split of the HopeEDI dataset.</li>
    </ul>
    
  </li>
  <li>Create a virtual enviroment and run all .py scripts. All packages were on their latests version (may 2024) with the exception of torch, who was on its version 1.10.1.</li>
</ol>

## Computer Equipment Used

### Hardware Specifications

The experiments were performed with the follow hardware:
<ul>
    <li>Graphic card: NVIDIA Quadro RTX 6000/8000</li>
    <li>Processor: Intel Xeon E3-1200</li>
    <li>RAM: 62 gb</li>
    <li>VRAM: 46 gb</li>
</ul>


### Software Specifications

The employed software was the follow:
<ul>
    <li>CUDA  V10.1.243</li>
    <li>OS: Ubuntu Server 20.04.3 LTS</li>
    <li>Vim version 8.1</li>
    <li>Python version: 3.9.5</li>
</ul>
