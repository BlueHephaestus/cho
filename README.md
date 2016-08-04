##CHO - Cool Hyper-Parameter Optimizer

What started with [Dennis](https://github.com/DarkElement75/dennis), I have now moved to it's own repository for future development. By default, it should be much better than grid search at finding ideal hyper parameters for the given network topology, and with good configuring(as well as more improvements of my own to make), I see no way it wouldn't be better than random grid search. I don't yet understand Bayesian Optimization yet, but I am fairly certain this is not better than that! Given a range of values and configuration settings, as well as necessary topology information initially in the cho_config.py file, CHO will do the following:

1. Generate vectors of Hyper Parameter(HP) values
2. Get cartesian product
3. Get average point output for each HP in the Cartesian Product i.e. if three learning rates 0.3, 0.4, 0.5 and three mini batch sizes 10, 20, 30, we'd get the three values for (10, 0.3), (10, 0.4), (10, 0.5) and average these outputs since the effect of m=10 should be equivalent across the different learning rates. This gives us one value for our m=10(and the rest of the mini batch sizes) to use in the following steps. Note: I realize m is a relatively independent HP and can be determined as such.
4. Use these new points for each HP to generate a quadratic linear regression of the form A + Bx + Cx^2 for each HP
5. Add all of our quadratic linear regressions together to get a multivariable equation such as 5 + 4m + 7.2m^2 + .32n + 6.32n^2
6. Compute the minimum to 1e-8 of our multivariable function using scipy.optimize.minimize with respective bounds
7. Use our new minimum HP values to generate new ranges after decreasing step size, if step size is under our threshold then stop for this HP
8. Go to Number 1 until all HPs are done

This has shown to be way faster and better than me, in fact what I thought was a bug at once was actually the most efficient mini batch size I could have obtained, and it just knew better than me. While it is still in constant development and improvement, it determined the ideal(at least I hope they were ideal) HPs for a shallow network topology (2209-100-30-7) I used with DENNIS earlier, and is now what I use for optimizing HPs for any of my experiments where I deem that necessary. Upon comparing with my own hypothesis for the best values, it gained a 20% improvement in validation accuracy, the metric upon which I chose it to search them with(although this can be REALLY easily changed, I chose validation accuracy because I see that as most relevant. I have considered a combination of accuracies for it to look at, but I'll stick with this until I see a reason that justifies changing it). 

Note: I may implement the MK system if I make enough improvements. In which case, this is Mk. 1
