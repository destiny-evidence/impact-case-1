# 

## Database connection

To access data from NACSOS, you will need to set up a tunnel to map port 5433 on your own machine to port 5432 on the remote machine. This will only work if you have a PIK account and access to the NACSOS host machine.

```
ssh -N -L 5433:localhost:5432 se164 -J ts01
```