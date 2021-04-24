#include "hyb_reduc.h"

#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void shared_reduc_init(shared_reduc_t *sh_red, int nthreads, int nvals)
{
  /* A COMPLETER */
  sh_red->nvals = nvals;
  sh_red->nthreads = nthreads;

  sh_red->red_val = malloc(sizeof(double) * nvals);
  memset(sh_red->red_val, 0, sizeof(double) * nvals);
  
  sh_red->red_mut = malloc(sizeof(pthread_mutex_t));
  pthread_mutex_init(sh_red->red_mut, NULL);

  sh_red->red_bar = malloc(sizeof(pthread_barrier_t));
  pthread_barrier_init(sh_red->red_bar, NULL, nthreads);

  sh_red->sem = malloc(sizeof(sem_t));
  
  sem_init(sh_red->sem, 0, 0);


  sh_red->master = 0;
}

void shared_reduc_destroy(shared_reduc_t *sh_red)
{
  /* A COMPLETER */
  free(sh_red->sem);
  
  pthread_barrier_destroy(sh_red->red_bar);
  free(sh_red->red_bar);

  pthread_mutex_destroy(sh_red->red_mut);
  free(sh_red->red_mut);

  free(sh_red->red_val);
}

/*
 * Reduction  hybride MPI/pthread
 * in  : tableau des valeurs a reduire (de dimension sh_red->nvals)
 * out : tableau des valeurs reduites  (de dimension sh_red->nvals)
 */
void hyb_reduc_sum(double *in, double *out, shared_reduc_t *sh_red)
{

  int principal = 0;

//chaque thread va rentrer dans le mutex et faire la somme dans red_val[i] qui est partagé par tout les threads d'un processus
  pthread_mutex_lock(sh_red->red_mut);
  {


     if (sh_red->master == 0)
      {
       //le thread master va mettre la valeur de la variable principal à 1
        principal = 1;
        sh_red->master = 1;
        

      }
      
    for (int i = 0; i < sh_red->nvals; i++)
      {         
        sh_red->red_val[i] += in[i];
      }
   
  }
  pthread_mutex_unlock(sh_red->red_mut);

  //on atend le dernier thread
  pthread_barrier_wait(sh_red->red_bar);

  
  if (principal == 1) 
    {

      int root = 0; 
        
      int rank = 0;
      /* rang de processus*/
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      int size = 0;
      /* le nombre de processus*/
      MPI_Comm_size(MPI_COMM_WORLD, &size);


      double *buff = malloc(sizeof(double) * size);


      for (int i = 0; i < sh_red->nvals; i++)
        { 
         //le thread master du processus 0 va collecter tout les tableaux red_val[i] des autres threads master d'audtres processus et les mettre dans buff
          MPI_Gather(&(sh_red->red_val[i]), 1, MPI_DOUBLE, buff, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

          if (rank == root)
            {
              for (int j = 0; j < size; j++)
                {
                  out[i] += buff[j];
                // le thread master du processus 0 va mettre à jour sa  valeur du tableau red_val
                  sh_red->red_val[i] = out[i];
                }

            }          

        }




      free(buff);
      //le thread maitre de processus 0 va diffuser les valeurs du tableau red_val[i] aux autres processus
      MPI_Bcast( sh_red->red_val, sh_red->nvals, MPI_DOUBLE, root, MPI_COMM_WORLD);
     //le thread maitre de processus 0 va diffuser les valeurs du tableau out[i] aux autres processus
      MPI_Bcast(out, sh_red->nvals, MPI_DOUBLE, root, MPI_COMM_WORLD);

  
    }
  /* si je ne suis pas le thread maitre */
  else
    {

      sem_wait(sh_red->sem);

      for (int i = 0; i < sh_red->nvals; i++)
        { 

       //mettre à jour les valeurs du tableau out[i] pour les autres threads de tout les processus
          out[i] = sh_red->red_val[i];
        }

     
    }

      sem_post(sh_red->sem);
    
}







