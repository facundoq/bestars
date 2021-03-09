J/ApJS/161/118    byH{alpha} photometry in open clusters     (McSwain+, 2005)
================================================================================
The evolutionary status of Be stars: results from a photometric study of
southern open clusters.
    McSwain M.V., Gies D.R.
   <Astrophys. J. Suppl. Ser., 161, 118-146 (2005)>
   =2005ApJS..161..118M
================================================================================
ADC_Keywords: Clusters, open ; Stars, Be ; Stars, emission ; Photometry, uvby
Keywords: open clusters and associations: individual (Basel 1, Bochum 13,
          Collinder 272, Haffner 16, Hogg 16, Hogg 22, IC 2395, IC 2581,
          IC 2944, NGC 2343, NGC 2362, NGC 2367, NGC 2383, NGC 2384, NGC 2414,
          NGC 2421, NGC 2439, NGC 2483, NGC 2489, NGC 2571, NGC 2659, NGC 3293,
          NGC 3766, NGC 4103, NGC 4755, NGC 5281, NGC 5593, NGC 6178, NGC 6193,
          NGC 6200, NGC 6204, NGC 6231, NGC 6249, NGC 6250, NGC 6268, NGC 6322,
          NGC 6425, NGC 6530, NGC 6531, NGC 6604, NGC 6613, NGC 6664,
          Ruprecht 79, Ruprecht 119, Ruprecht 127, Ruprecht 140, Stock 13,
          Stock 14, Trumpler 7, Trumpler 18, Trumpler 20, Trumpler 27,
          Trumpler 28, Trumpler 34, vdB-Hagen 217) - stars: emission-line, Be

Abstract:
    Be stars are a class of rapidly rotating B stars with circumstellar
    disks that cause Balmer and other line emission. There are three
    possible reasons for the rapid rotation of Be stars: they may have
    been born as rapid rotators, spun up by binary mass transfer, or spun
    up during the main-sequence (MS) evolution of B stars. To test the
    various formation scenarios, we have conducted a photometric survey of
    55 open clusters in the southern sky. Of these, five clusters are
    probably not physically associated groups and our results for two
    other clusters are not reliable, but we identify 52 definite Be stars
    and an additional 129 Be candidates in the remaining clusters.

Description:
    We made photometric observations of the clusters over 14 nights
    between 2002 March 27-April 2 and 2002 June 1925 with the CTIO (Cerro
    Tololo Inter-American Observatory) 0.9m telescope and SITe 2048 CCD.

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
table1.dat       114    17541   Photometry of selected stars in open clusters
table2.dat        70       55   Summary of open clusters
refs.dat          98       34   References
table4.dat        37       48   Distribution of open clusters
--------------------------------------------------------------------------------

See also:
  J/ApJ/622/1052 : byH{alpha} photometry of NGC 3766 (McSwain+, 2005)
  J/A+A/402/549  : UBVRIH{alpha} photometry of NGC 3293 (Baume+, 2003)
  VII/229        : Optically visible open clusters and Candidates
                                                    (Dias+ 2002-2005)
  J/A+A/373/153  : CCD {Delta}a-photometry of 5 open clusters (Paunzen+, 2001)
  J/A+A/360/529  : BVI photometry of 4 young open clusters (Piatti+, 2000)
  J/A+A/370/931  : BVI photometry of 4 open clusters (Piatti+, 2001)
  J/A+A/369/511  : Open star clusters. III. NGC 4103, 5281, 4755 (Sanner+, 2001)
  J/A+AS/124/13  : UBVRI photometry of the cluster Collinder 272 (Vazquez+ 1997)

Byte-by-byte Description of file: table1.dat
--------------------------------------------------------------------------------
   Bytes Format Units     Label     Explanations
--------------------------------------------------------------------------------
   1- 13  A13   ---       Cluster   Cluster name
  15- 18  I4    ---       MG        Star identifying number (1)
  20- 21  I2    h         RAh       Hour of Right ascension (J2000.0)
  23- 24  I2    min       RAm       Minute of Right ascension (J2000.0)
  26- 30  F5.2  s         RAs       Second of Right ascension (J2000.0)
      32  A1    ---       DE-       Sign of the Declination (J2000.0)
  33- 34  I2    deg       DEd       Degree of Declination (J2000.0)
  36- 37  I2    arcmin    DEm       Arcminute of Declination (J2000.0)
  39- 42  F4.1  arcsec    DEs       Arcsecond of Declination (J2000.0)
  44- 49  F6.3  mag       ymag      Stromgren y magnitude
  51- 55  F5.3  mag     e_ymag      1{sigma} error in ymag
  57- 62  F6.3  mag       b-y       Stromgren b-y color index
  64- 68  F5.3  mag     e_b-y       1{sigma} error in b-y
  70- 75  F6.3  mag       y-Ha      y-H{alpha} color index
  77- 81  F5.3  mag     e_y-Ha      1{sigma} error in y-Ha
  83- 85  A3    ---       Code      Code for type (2)
  87- 93  I7    ---       Webda     ? The WEBDA identification number
  95-114  A20   ---       OName     Other identification
--------------------------------------------------------------------------------
Note (1): Star identified as Cl* "Cluster" MG "MG",
          Cl* VDBH 217 MG NNN for "vdB-Hagen 217" cluster.
Note (2): Code for type, defined as follows:
     B = B-type star;
    Be = definite Be star;
   Be? = possible Be star;
     O = other stars in the field;
     F = probable foreground star.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table2.dat
--------------------------------------------------------------------------------
   Bytes Format Units    Label     Explanations
--------------------------------------------------------------------------------
   1- 13  A13   ---      Cluster   Cluster name
      14  A1    ---    q_Cluster   Reliability of cluster parameters (1)
  16- 17  I2    h        RAh       Hour of Right Ascension (J2000.0)
  19- 20  I2    min      RAm       Minute of Right Ascension (J2000.0)
  22- 23  I2    s        RAs       Second of Right Ascension (J2000.0)
      25  A1    ---      DE-       Sign of the Declination (J2000.0)
  26- 27  I2    deg      DEd       Degree of Declination (J2000.0)
  29- 30  I2    arcmin   DEm       Arcminute of Declination (J2000.0)
  32- 33  I2    arcsec   DEs       Arcsecond of Declination (J2000.0)
  35- 38  F4.2  mag      E(b-y)    Color excess
  40- 44  F5.2  mag      V0-MV     Distance modulus
  46- 49  F4.2  [yr]     logAge    Log of age
      51  I1    ---      Be        Number of definite Be stars
  53- 54  I2    ---      Be?       Number of possible Be stars
      56  A1    ---    l_B+Be      [>] Lower limit flag on N(B*)
  57- 59  I3    ---      B+Be      Number of B and Be stars identified
  61- 70  A10   ---      Ref       References, detailed in refs.dat
--------------------------------------------------------------------------------
Note (1): Flag on Cluster, defined as follows:
      a = Cluster parameters unreliable.
      b = May not be a true cluster.
      c = May contain PMS Herbig Be stars.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: refs.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1-  2  I2    ---     Ref       Reference number
   4- 22  A19   ---     Bibcode   Bibcode
  24- 42  A19   ---     Aut       Author's name
  44- 98  A55   ---     Com       Comment
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table4.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 13  A13   ---     Cluster   Cluster name
  15- 20  F6.2  deg     GLON      Galactic longitude
  22- 26  F5.2  deg     GLAT      Galactic latitude
  28- 31  F4.2  kpc     Dist      Distance to the Sun
  33- 37  F5.2  kpc     DistGC    Distance to the Galactic Center
--------------------------------------------------------------------------------

History:
    From electronic version of the journal

References:
  McSwain & Gies, Paper I    2005ApJ...622.1052M, Cat. J/ApJ/622/1052
================================================================================
(End)                   Greg Schwarz [AAS], Marianne Brouty [CDS]    12-Jul-2006
