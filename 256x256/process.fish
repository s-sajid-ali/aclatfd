#!/usr/bin/env fish

for i in (seq 0 20)
  if test $i -le 9
    python ../scripts/import_charge_dist.py /home/sajid/data/accelsim/onebunchff.02/turn_particles_000$i.h5 ob2tp$i.npy
  else
    python ../scripts/import_charge_dist.py /home/sajid/data/accelsim/onebunchff.02/turn_particles_00$i.h5 ob2tp$i.npy
  end
end

for i in (seq 0 39)
  if test $i -le 9
    python ../scripts/import_charge_dist.py /home/sajid/data/accelsim/twobunchff.03/turn_particles_000$i.h5 tb2tp$i.npy
  else
    python ../scripts/import_charge_dist.py /home/sajid/data/accelsim/twobunchff.03/turn_particles_00$i.h5 tb2tp$i.npy
  end
end
