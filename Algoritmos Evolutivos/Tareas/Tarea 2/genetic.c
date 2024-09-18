#define popsize         100             /* Population size         */
#define codesize        16              /* Number of bits used to
					   represent an individual */
#include <stdio.h>
#include <math.h>
#include "genetic.h"

main()
{
	struct individual *temp;
	float Rseed;

	cls();
	printf("Enter the mutation rate -------------------------> ");
	scanf("%f",&Pm);
	printf("Enter the probability of crossover --------------> ");
	scanf("%f",&Pc);
	printf("Enter the maximum number of generations ---------> ");
	scanf("%d",&Gmax);
	do
	{
	printf("Enter random number seed, 0.0 to 1.0 ------------> ");
	scanf("%f", &Rseed);
	}
	while((Rseed < 0.0) || (Rseed > 1.0));

	/* Perform the previous memory allocation */

	memory_allocation();

	/* Initialize random number generator */

	randomize();

	nmutation = 0;
	ncross = 0;
	bestfit.fitness = 0.0;
	bestfit.generation = 0;

	/* Initialize the populations and report statistics */
	initialize_pop();
	statistics(oldpop);
	initial_report();

	for (gen=0; gen<Gmax; gen++)  /* Iterate the given number of generations */
	{

	/* Create a new generation */

		generation();

	/* Compute fitness statistics on new populations */

		statistics(newpop);

	/* Report results for new generation */

		report();

	/* Advance the generation */

	temp = oldpop;
	oldpop = newpop;
	newpop = temp;

	}

	free_all();
}

/* Define randomly the population for generation # 0 */

void initialize_pop()
{
	int j, k;

	for (j=0; j < popsize; j++)
	  for (k=0; k < codesize; k++)
	   {
		oldpop[j].chrom[k]=flip(0.5);
		oldpop[j].x=decode(oldpop[j].chrom, codesize);
		objfunc(&(oldpop[j]));
		oldpop[j].parent[0] = 0;        /* Initialize parent info */
		oldpop[j].parent[1] = 0;
		oldpop[j].xsite1 = 0;           /* Initialize crossover sites */
		oldpop[j].xsite2 = 0;
		oldpop[j].placemut = 0;         /* Initialize mutation place */
		oldpop[j].mutation = 0;         /* Initialize mutation indicator */

		}

}

/* Print the initial report of the parameters given by the user */

void initial_report()
{
	printf("\n      Parameters used with the Genetic Algorithm \n");
	printf(" Total population size          =       %d\n",popsize);
	printf(" Chromosome length              =       %d\n",codesize);
	printf(" Maximum number of generations  =       %d\n",Gmax);
	printf(" Crossover probability          =       %f\n",Pc);
	printf(" Mutation probability           =       %f\n",Pm);
	printf("\n\n");
}

/* Perform the memory allocation needed for the strings that will be generated */

void memory_allocation()
{
	unsigned numbytes;
	char *malloc();
	int i;

	/* Allocate memory for old and new populations of individuals */

	numbytes = popsize*sizeof(struct individual);

	if ((oldpop = (struct individual *) malloc(numbytes)) == NULL)
		nomemory("old population");

	if ((newpop = (struct individual *) malloc(numbytes)) == NULL)
		nomemory("new population");

	/* Allocate memory for chromosome strings in populations */

	for (i=0; i< popsize; i++) {

		if ((oldpop[i].chrom = (unsigned *) malloc(numbytes)) == NULL)
			nomemory("old population chromosomes");

		if ((newpop[i].chrom = (unsigned *) malloc(numbytes)) == NULL)
			nomemory("new population chromosomes");

		if ((bestfit.chrom = (unsigned *) malloc(numbytes)) == NULL)
			nomemory("bestfit chromosomes");
	}

	memory_for_selection();
}


/* Allocate memory for performing tournament selection */

void memory_for_selection()
{
	unsigned numbytes;
	char *malloc();

	numbytes = popsize*sizeof(int);
	if ((tournlist = (int *) malloc(numbytes)) == NULL)
		nomemory("tournament list");

	tournsize = 2;  /* Use binary tournament selection */
}

/* When done, free all memory */

void free_all()
{
	int i;

	for (i=0; i < popsize; i++) {

		free(oldpop[i].chrom);
		free(newpop[i].chrom);
	}

	free(oldpop);
	free(newpop);

	free_selection();
}

/* Free memory used for selection */

void free_selection()
{
	free(tournlist);
}

/* Notify if we run out of memory when generating a chromosome */

void nomemory(char *string)
{
	printf("ERROR!! --> malloc: out of memory making %s\n",string);
	exit(1);
}

/* Calculate population statistics */

void statistics(struct individual *pop)
{
	int i,j;

	sumfitness = 0.0;
	min = pop[0].fitness;
	max = pop[0].fitness;

	/* Loop for max, min, sumfitness */

	for (j=0; j < popsize; j++) {

		sumfitness = sumfitness + pop[j].fitness;            /* Accumulate */
		if (pop[j].fitness > max) { max = pop[j].fitness;    /* New maximum */
					    localmax = pop[j].chrom; /* Store string */
					  }

		if (pop[j].fitness < min) min = pop[j].fitness;     /* New minimum */

	/* Define new global best-fit individual */

	if (pop[j].fitness > bestfit.fitness) {

		for (i=0; i < codesize; i++)

		  bestfit.chrom[i] = pop[j].chrom[i];
		  bestfit.fitness = pop[j].fitness;
		  bestfit.generation = gen;
		}
	}

	/* Calculate average */

	avg = sumfitness/popsize;
}

/* Create next batch of 55 random numbers */

void advance_random()

{
    int j1;
    double new_random;

    for(j1 = 0; j1 < 24; j1++)
    {
	new_random = oldrand[j1] - oldrand[j1+31];
	if(new_random < 0.0) new_random = new_random + 1.0;
	oldrand[j1] = new_random;
    }

    for(j1 = 24; j1 < 55; j1++)
    {
	new_random = oldrand [j1] - oldrand [j1-24];
	if(new_random < 0.0) new_random = new_random + 1.0;
	oldrand[j1] = new_random;
    }

}

/* Flip a biased coin - true if heads */

int flip(float prob)

{
    float randomperc();

    if(randomperc() <= prob)
	return(1);
    else
	return(0);

}

/* initialization routine for randomnormaldeviate */

void initrandomnormaldeviate()

{
    rndcalcflag = 1;
}

/* normal noise with specified mean & std dev: mu & sigma */

double noise(double mu,double sigma)

{
    double randomnormaldeviate();

    return((randomnormaldeviate()*sigma) + mu);
}

/* Initialize random numbers batch */

void randomize()

{
    int j1;

    for(j1=0; j1<=54; j1++)
      oldrand[j1] = 0.0;

    jrand=0;

     warmup_random(Rseed);
}

/* random normal deviate after ACM algorithm 267 / Box-Muller Method */

double randomnormaldeviate()

{
    double sqrt(), log(), sin(), cos();
    float randomperc();
    double t, rndx1;

    if(rndcalcflag)
    {
	rndx1 = sqrt(- 2.0*log((double) randomperc()));
	t = 6.2831853072 * (double) randomperc();
	rndx2 = sin(t);
	rndcalcflag = 0;
	return(rndx1 * cos(t));
    }

    else

    {
	rndcalcflag = 1;
	return(rndx2);
    }

}

/* Fetch a single random number between 0.0 and 1.0 - Subtractive Method */
/* See Knuth, D. (1969), v. 2 for details */
/* name changed from random() to avoid library conflicts on some machines*/

float randomperc()

{
    jrand++;

    if(jrand >= 55)
    {
	jrand = 1;
	advance_random();
    }

    return((float) oldrand[jrand]);
}

/* Pick a random integer between low and high */

int rnd(int low, int high)

{
    int i;
    float randomperc();

    if(low >= high)
	i = low;

    else

    {
	i = (randomperc() * (high - low + 1)) + low;
	if(i > high) i = high;
    }

    return(i);
}

/* real random number between specified limits */

float rndreal(float lo ,float hi)

{
    return((randomperc() * (hi - lo)) + lo);
}

/* Get random off and running */

void warmup_random(float random_seed)

{
    int j1, ii;
    double new_random, prev_random;

    oldrand[54] = random_seed;
    new_random = 0.000000001;
    prev_random = random_seed;

    for(j1 = 1 ; j1 <= 54; j1++)

    {
	ii = (21*j1)%54;
	oldrand[ii] = new_random;
	new_random = prev_random-new_random;
	if(new_random<0.0) new_random = new_random + 1.0;
	prev_random = oldrand[ii];
    }

    advance_random();
    advance_random();
    advance_random();

    jrand = 0;

}

/* Perform all the work involved with the creation of a new generation of chromosomes */

void generation()
{

  int mate1, mate2, jcross1, jcross2, j = 0;

  /* perform any preselection actions necessary before generation */

  preselect();

  /* select, crossover, and mutation */

  do
    {
      /* pick a pair of mates */

      mate1 = select();
      mate2 = select();

      /* Crossover and mutation */

      crossover(oldpop[mate1].chrom, oldpop[mate2].chrom,newpop[j].chrom,
		newpop[j+1].chrom,&jcross1,&jcross2);

      mutation(&(newpop[j]));
      mutation(&(newpop[j+1]));

      /* Decode string, evaluate fitness, & record */
      /* parentage date on both children */

      newpop[j].x=decode(newpop[j].chrom, codesize);
      newpop[j+1].x=decode(newpop[j+1].chrom, codesize);

      objfunc(&(newpop[j]));
      newpop[j].parent[0] = mate1+1;
      newpop[j].xsite1 = jcross1;
      newpop[j].xsite2 = jcross2;
      newpop[j].parent[1] = mate2+1;
      objfunc(&(newpop[j+1]));
      newpop[j+1].parent[0] = mate1+1;
      newpop[j+1].xsite1 = jcross1;
      newpop[j+1].xsite2 = jcross2;
      newpop[j+1].parent[1] = mate2+1;

      /* Increment population index */

      j = j + 2;
    }

  while(j < (popsize-1));

}

/* Perform a preselection */

void preselect()
{
	reset();
	tournpos = 0;
}

/* Implementation of a tournament selection process */

int select()
{
	int pick, winner, i;

	/* If remaining members not enough for a tournament, then reset list */

	if ((popsize - tournpos) < tournsize)
	{
		reset();
		tournpos = 0;
	}

	/* Select tournament size structures at random and conduct a tournament */

	winner = tournlist[tournpos];

	for (i=1; i < tournsize; i++)
	{
		pick=tournlist[i+tournpos];
		if(oldpop[pick].fitness > oldpop[winner].fitness) winner=pick;
	}

	/* Update tournament position */

	tournpos += tournsize;
	return(winner);

}

/* Shuffle the tournament list at random */

void reset()
{
	int i, rand1, rand2, temp;

	for (i=0; i < popsize; i++) tournlist[i]=i;

	for (i=0; i < popsize; i++)
	{
		rand1=rnd(i,popsize-1);
		rand2=rnd(i,popsize-1);
		temp=tournlist[rand1];
		tournlist[rand1]=tournlist[rand2];
		tournlist[rand2]=temp;
	}

}

/* Perform a mutation in a random string, and keep track of it */

void mutation(struct individual *ind)

{
	int placemut,k;

	/* Do mutation with probability Pm */

	if (flip(Pm)) {
			placemut=rnd(0,(codesize-1));
			ind->placemut=codesize-placemut-1;
			ind->mutation=1;
			nmutation++;
			for (k=0;k<codesize;k++) {

				if (placemut==k) {
					if ((ind->chrom[k])==0) ind->chrom[k]=1;
					else ind->chrom[k]=0;
				}
			}
		     }

	else ind->mutation=0;

}

/* Cross 2 parent strings, place in 2 child strings */

void crossover (unsigned *parent1, unsigned *parent2,
	       unsigned *child1, unsigned *child2,
	       int *jcross1, int *jcross2)

{
    int j;
    unsigned temp;

    /* Do crossover with probability Pc */

    if(flip(Pc))
    {
	*jcross1 = rnd(0,(codesize - 1));          /* Crosspoint 1 between 1 and length-1 */

	/* Define crosspoint 2 between jcross1+1 and length-1 */

	if (*jcross1 < (codesize-1)) *jcross2 = rnd(*jcross1+1,(codesize - 1));

	else {     *jcross2=rnd(0,(*jcross1-1)); /* Make sure that jcross1 is */
		   temp=*jcross1;                /* always less than jcross2  */
		   *jcross1=*jcross2;
		   *jcross2=temp;
	     }

	ncross++;               /* Keep track of the number of crossovers */

	for (j=(codesize-1); j>=(codesize - *jcross1); j--)
		{
		child1[j]=parent1[j];
		child2[j]=parent2[j];
		}
	for (j=(codesize - *jcross1)-1; j>=(codesize - *jcross2); j--)
		{
		child1[j]=parent2[j];
		child2[j]=parent1[j];
		}
	for (j=(codesize - *jcross2)-1; j>=0; j--)
		{
		child1[j]=parent1[j];
		child2[j]=parent2[j];
		}
     }
	else {
		for (j=0; j<=(codesize-1); j++) {
		  child1[j]=parent1[j];
		 child2[j]=parent2[j];
		}
		*jcross1=0; *jcross2=0;
	}

}

/* Write the population report on the screen */

void report()

{
	printf("\n_____________________________________________________");
	printf("________________________________________________________\n");
	printf("                                      P O P U L A T I O N          R E P O R T  \n");
	printf(" Generation # %3d                             Generation # %3d\n",gen,(gen+1));
	printf(" num      string           x       fitness    parents    x1   x2  mut ");
	printf("place     string        x       fitness\n");
	printf("_____________________________________________________");
	printf("________________________________________________________\n");

	writepop();

	printf("_____________________________________________________");
	printf("________________________________________________________\n");

    /* write the summary statistics in global mode  */

	printf("Generation # %d Accumulated Statistics: \n",gen);
	printf("Total Crossovers = %d, Total Mutations = %d\n", ncross,nmutation);
	printf("min = %f   max = %f   avg = %f   sum = %f\n",min,max,avg,sumfitness);
	printf("Best individual in this generation = ");
	writechrom(localmax); printf("\n");
	printf("Global Best Individual so far, Generation # %d:\n",bestfit.generation);
	printf("String = ");
	writechrom((&bestfit)->chrom);
	printf("    Fitness = %f\n", bestfit.fitness);

 }

/* Give the appropriate format to the several values that get printed */

void writepop()
{
    struct individual *pind;
    int j;

    for(j=0; j<popsize; j++)
    {
	printf("%3d)  ",j+1);

	/* Old string */

	pind = &(oldpop[j]);
	writechrom(pind->chrom);
	printf(" %10.4f  %5.4f | ", pind->x, pind->fitness);

	/* New string */

	pind = &(newpop[j]);
	printf("(%3d,%3d)  %3d   %3d  ",
	pind->parent[0], pind->parent[1], pind->xsite1, pind->xsite2);
	if (pind->mutation==1)  printf(" Y %3d  ",pind->placemut);
	else printf(" N  --- ");
	writechrom(pind->chrom);
	printf(" %10.4f  %5.4f", pind->x, pind->fitness);
	printf("\n");
    }
}

/* Print out each chromosome from the most significant bit to the least significant bit */

void writechrom(unsigned *chrom)

{
	int j;

	for (j=(codesize-1); j>=0; j--) {
		if (chrom[j]==0) printf("0");
			else printf("1");
	}

}

/* Use a system call to clear the screen */

void cls()

{
	system("clear");
}

/* Decode the chromosome string into its corresponding decimal value */

double decode(unsigned *chrom, int length)
{
	int j;
	double accum=0.0;

	for (j=0; j<length; j++) {
		if (chrom[j]==1) accum+=pow(2.0,(double)j);
		}

	return accum;

}

/* Define the objective function. In this case, f(x) = x^2 */

void objfunc(struct individual *ind)

{
	double coef;

	ind->fitness=0.0;

	coef=pow(2.0, (double)codesize) - 1.0;  /* Normalize the function */

	ind->fitness=pow((ind->x)/coef, 2.0);
	ind->fitness=((int)((ind->fitness)*10000.0+0.5))/10000.0;

}
