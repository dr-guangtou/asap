"""Utilities for A.S.A.P model."""

import pickle


__all__ = ["mcmc_save_chains", "mcmc_save_results", "mcmc_load_chains"]


def mcmc_save_chains(mcmc_chain_file, mcmc_chain):
    """Pickle the chain to a file."""
    pickle_file = open(mcmc_chain_file, 'wb')
    pickle.dump(mcmc_chain, pickle_file)
    pickle_file.close()

    return None


def mcmc_save_results(pkl_name, mcmc_result):
    """Pickle the MCMC results to a file."""
    pkl_file = open(pkl_name, 'wb')
    mcmc_position, mcmc_prob, mcmc_state = mcmc_result
    pickle.dump(mcmc_position, pkl_file, -1)
    pickle.dump(mcmc_prob, pkl_file, -1)
    pickle.dump(mcmc_state, pkl_file, -1)
    pkl_file.close()

    return None


def mcmc_load_chains(mcmc_chain_file):
    """Load the pickled chain."""
    pickle_file = open(mcmc_chain_file, 'rb')
    mcmc_chain = pickle.load(pickle_file)
    pickle_file.close()

    return mcmc_chain
