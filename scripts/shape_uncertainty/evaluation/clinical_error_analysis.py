import numpy as np

from scripts.shape_uncertainty.simulated_data.shape_ground_truth import (
    get_analytic_shape_summary,
    compute_critical_point,
)

from scripts.shape_uncertainty.shape_extraction.shape_uncertainty import (
    compute_shape_uncertainty, medoid_summary)



def has_rebound(summary):
    """Check whether a Wilkerson DGP shape summary turns from decreasing to increasing.

    This is read from thr ground truth shape summary. 

    Args:
        summary: list of tuples (str, float); a shape summary as (state,
            start_time) pairs, sorted chronologically.

    Returns:
        bool; whether any increasing state is preceded by a decreasing one.
    """
    # An increasing state that follows a decreasing one, rather than one
    # particular pair, so that decreasing -> constant -> increasing counts.
    seen_decreasing = False
    for state, _ in summary:
        if state.endswith("decreasing"):
            seen_decreasing = True
        elif state.endswith("increasing") and seen_decreasing:
            return True
    return False

class ClinicalErrorEvaluator:
    """This class owns the design of the clinical error experiment.

    We ask three questions about the rebound, the clinically consequential shape.

        Question 1: What does the point forecast get wrong, and on whom? Among
            individuals whose ground truth curve rebounds,
            - for what share does the point forecast call the trajectory
              monotone, so that the error is made silently?
            - where on the horizon do those turns occur, and how well were those
              individuals fitted?

        Question 2: Does reporting the ensemble's consensus instead repair it?
            Comparing the two point answers cell by cell,
            - how many missed rebounds does the consensus recover, and how many
              correct calls does it lose in exchange?
            - how many spurious rebounds does it retract, and how many new ones
              does it introduce?

        Question 3: Does the disagreement inside the cloud find what neither
            point answer reports? Among the individuals the point forecast calls
            monotone,
            - do the share of the cloud asserting a rebound, or the shape
              uncertainty U alone, rank the missed rebounds above the correctly
              monotone ones.

    Args:
        test: SimulatedDataset; the test split every index refers to.
        uncertainty_engine: UncertaintyEngine; used only to rebuild a single
            individual's cloud where the result does not carry one, which is the
            case for a combined result, whose clouds are too large to hold.
        result: dict as returned by
            UncertaintyEngine.predict_with_epistemic_uncertainty or by
            predict_with_combined_uncertainty, supplying "U", and either
            "consensus" and "shapes" or the "coefficients" they rebuild from.
        point_forecast_engine: InferenceEngine; the single trained model whose
            mean trajectory is the deterministic forecast under test. This is
            what a practitioner reports without any uncertainty machinery at all.
        shape_config: dict; the extraction thresholds the ground-truth summaries
            are read under, so that truth and forecast are summarised on the
            same terms.

    Attributes:
        test, uncertainty_engine, result, point_forecast_engine, shape_config:
            as passed.
    """

    def __init__(self, test, uncertainty_engine, result, point_forecast_engine,
                 shape_config):
        self.test = test
        self.uncertainty_engine = uncertainty_engine
        self.result = result
        self.point_forecast_engine = point_forecast_engine
        self.shape_config = shape_config

        self._true_summaries = None
        self._point_summaries = None
        self._clouds = {}
        self._true_flags = None
        self._point_flags = None
        self._consensus_flags = None

    # ----------------------- Reading shape summaries ----------------------

    def _get_true_shape_summaries(self):
        """Return the analytic shape summary of every individual's curve.

        Returns:
            list of length D; element i is that individual's shape summary as a
                list of (state, start_time) tuples.
        """
        if self._true_summaries is None:
            self._true_summaries = get_analytic_shape_summary(
                self.test,
                self.shape_config["upsilon_rel_1"],
                self.shape_config["upsilon_rel_2"],
                self.shape_config["upsilon_rel_prune"],
                self.shape_config["do_prune"],
            )
        return self._true_summaries

    def _get_point_forecast_summaries(self):
        """Return the shape summary the deterministic point forecast reports.
        
        The remainder of this class is meant for analysing failures of the shape
        summary returned by a single model as they are obtained.

        Returns:
            list of length D; element i is the point forecast's shape summary
                for that individual.
        """
        if self._point_summaries is None:
            self._point_summaries = (
                self.point_forecast_engine.predict_mean_shape_summary(self.test.X))
        return self._point_summaries

    def _get_individual_cloud(self, index):
        """Return one individual's cloud of shape summaries.

        There are two cases:
            Case 1: Ensemble. 
                - Here the result fits in memory and already and we can read off
                the shape summary form the result.
            Case 2: Nested cloud. 
                - Here the shape summaries do not fit in memory and we need to
                rebuild the shape summaries from the saved coefficent draws. 

        Args:
            index: int; the individual to rebuild.

        Returns:
            list of shape summaries, one per member of that individual's cloud.
        """
        # Case 1: If we computed the result for this individual before, we return it. 
        if index in self._clouds:
            return self._clouds[index]

        # Case 2: If not computed before and the saved results includes a shape key
        if "shapes" in self.result:
            self._clouds[index] = self.result["shapes"][index]
            return self._clouds[index]

        # Case 3: If not computed before and the save results does not include a shape key
        summaries = []
        for engine, block in zip(self.uncertainty_engine.engines,
                                 self.result["coefficients"]):
            summaries.extend(engine.predict_aleatoric_summaries(block[[index]])[0])

        self._clouds[index] = summaries
        return summaries


    def _get_individual_consensus(self, index):
        """Return the consensus shape summary the cloud reports for one individual.

        Args:
            index: int; the individual to read.

        Returns:
            list of tuples (str, float); the consensus shape summary.
        """
        # Case 1 - ensemble: In this case just return the consensus
        if "consensus" in self.result:
            return self.result["consensus"][index]

        # Case 1 - nested cloud: Compute the consensus summary
        summaries = self._get_individual_cloud(index)
        _, profile = compute_shape_uncertainty(summaries, self.test.T)
        consensus, _ = medoid_summary(summaries, self.test.T, profile)
        return consensus

    def get_shape_profile(self, index):
        """Return the regional shape uncertainty profile of one individual.

        Args:
            index: int; the individual to read.

        Returns:
            dict; the uncertainty profile, carrying "regions" and
                "region_uncertainty" among the keys compute_shape_uncertainty
                returns.
        """
        # Case 1 - ensemble: the result already carries every profile
        if "profiles" in self.result:
            return self.result["profiles"][index]

        # Case 2 - nested cloud: score this individual's rebuilt cloud
        summaries = self._get_individual_cloud(index)
        _, profile = compute_shape_uncertainty(summaries, self.test.T)

        return profile


    # ------------------------ Does the trajectory rebound --------------------------

    def get_true_rebound_flags(self):
        """For each individual, flag whether the ground truth rebounds.

        Returns:
            np.ndarray of shape (D,) and dtype bool.
        """
        if self._true_flags is None:
            # Get the summaries
            summaries = self._get_true_shape_summaries()

            # Check summariesf or a rebound
            rebound_flags = [has_rebound(summary) for summary in summaries]
            
            # Convert to boolean flags
            self._true_flags = np.array(rebound_flags, dtype=bool)

        return self._true_flags

    def get_point_forecast_rebound_flags(self):
        """For each individual, determine whether the point forecast predicts a rebound fo this indiviual.

        Returns:
            np.ndarray of shape (D,) and dtype bool.
        """
        if self._point_flags is None:
            # Get the summaries
            summaries = self._get_point_forecast_summaries()
            
            # Check summariesf or a rebound
            rebound_flags = [has_rebound(summary) for summary in summaries]
            
            # Convert to boolean flags
            self._point_flags = np.array(rebound_flags, dtype=bool)

        return self._point_flags


    def get_point_forecast_monotone_flags(self, slope):
        """For each individual, determine whether the point forecast carries one
        slope over the whole horizon.

        A rebound needs a rising state after a falling one, so a summary flagged
        here reports no rebound.

        Args:
            slope: str; "decreasing" or "increasing", the direction to require.

        Returns:
            np.ndarray of shape (D,) and dtype bool.
        """
        summaries = self._get_point_forecast_summaries()

        monotone_flags = [
            bool(summary) and all(state.endswith(slope) for state, _ in summary)
            for summary in summaries
        ]

        return np.array(monotone_flags, dtype=bool)


    def get_consensus_rebound_flags(self):
        """For each individual, determine whether the consensus predicts a rebound fo this indiviual.

        Returns:
            np.ndarray of shape (D,) and dtype bool.
        """
        if self._consensus_flags is None:
            
            # Determine rebound status for each individual's consensus
            rebound_flags = [
                has_rebound(self._get_individual_consensus(index))
                for index in range(self.test.D)
            ]
            
            # Convert to boolean flags
            self._consensus_flags = np.array(rebound_flags, dtype=bool)

        return self._consensus_flags

    # -------------------------- Get different samples of the test set-----------------------

    def get_rebounding_individuals(self):
        """Return the individuals whose generating curve rebounds.

        This function returns the individuals in the dataset that have a rebound. 

        Returns:
            np.ndarray of int; indices into the test split, ascending.
        """
        return np.flatnonzero(self.get_true_rebound_flags()) #  1D array of indices where the elements are not zero


    def get_missed_rebound_individuals(self):
        """Return the rebounding individuals reported as a monotone decline.

        This is the clinically consequential error, since a continuing decline
        reads as a treatment working.

        Returns:
            np.ndarray of int; indices into the test split, ascending.
        """
        # The truth turns and the forecast reports a decline throughout
        true_flags = self.get_true_rebound_flags()
        decline_flags = self.get_point_forecast_monotone_flags("decreasing")

        return np.flatnonzero(true_flags & decline_flags)


    def get_misreported_increase_individuals(self):
        """Return the rebounding individuals reported as a monotone increase.

        These are wrong too, but they deny the response rather than the relapse,
        and together with the missed rebounds and the detected ones they account
        for every rebounding individual.

        Returns:
            np.ndarray of int; indices into the test split, ascending.
        """
        true_flags = self.get_true_rebound_flags()
        increase_flags = self.get_point_forecast_monotone_flags("increasing")

        return np.flatnonzero(true_flags & increase_flags)


    def get_detected_rebound_individuals(self):
        """Return the rebounding individuals the point forecast reports as such.

        Returns:
            np.ndarray of int; indices into the test split, ascending.
        """
        true_flags = self.get_true_rebound_flags()
        point_flags = self.get_point_forecast_rebound_flags()

        return np.flatnonzero(true_flags & point_flags)


    def get_detection_population(self):
        """Return the individuals a warning would be issued to, and their labels.

        This function returns the population of all individuals
        where no rebound was predicted. 
        This is the population for which we may wish to issue a warning 
        abuout a potential rebound where the point forecast made a mnotone 
        prediction but where the uncertinty engine may predict otherwise.

        Returns:
            tuple (indices, labels):
                indices: np.ndarray of int; the individuals the point forecast
                    calls monotone, ascending.
                labels: np.ndarray of bool, aligned with indices; True where the
                    generating curve rebounds, i.e. the missed rebounds.
        """
        # Get the rebound flags for the point forecasts
        point_flags = self.get_point_forecast_rebound_flags()

        # Get the indices where there is NO point forecast rebound
        indices = np.flatnonzero(~point_flags)

        # Get the true rebound flags for the entire set
        all_true_flags = self.get_true_rebound_flags()

        # Filter the true flags to match the non-rebound indices
        subset_true_flags = all_true_flags[indices]

        return indices, subset_true_flags

   # ------------------  Get information on individuls -----------------------

    def get_shape_uncertainty(self, indices):
        """Return the shape uncertainty U of each individual.

        This function returns the shape uncertainty U for specified indiviuals. 

        Args:
            indices: sequence of int; the individuals to read.

        Returns:
            np.ndarray of shape (len(indices),) and dtype float; each in [0, 1],
                in the order the indices were given.
        """
        return np.asarray(self.result["U"], dtype=float)[np.asarray(indices, dtype=int)]

    def compute_rebound_support(self, indices):
        """Return the share of each cloud that asserts a rebound.

        Compute the share of a cloud of trajectories whose shape summary
        supports a rebound. Do this for every individual in "indices", respectively. 

        Args:
            indices: sequence of int; the individuals to score.

        Returns:
            np.ndarray of shape (len(indices),) and dtype float; each in [0, 1],
                in the order the indices were given.
        """
        int_indices = np.asarray(indices, dtype=int)
        mean_rebound_values = []

        for index in int_indices:
            # Extract the cloud of summaries
            cloud = self._get_individual_cloud(index)
            
            # Calculate the float mean in one readable line
            cloud_mean = float(np.mean([has_rebound(s) for s in cloud]))
            
            mean_rebound_values.append(cloud_mean)

        return np.array(mean_rebound_values)

    def compute_critical_points(self, indices):
        """Return the true turning point of each individual's generating curve.

        Args:
            indices: sequence of int; the individuals to read.

        Returns:
            np.ndarray of shape (len(indices),) and dtype float; 
                the critical points the order the indices were given;
                np.nan where the curve has no real stationary point, or falls
                outside [0, T]
        """
        int_indices = np.asarray(indices, dtype=int)

        return np.array([
            compute_critical_point(self.test.params["g"][index],
                                   self.test.params["d"][index],
                                   self.test.params["rho"][index])
            for index in int_indices])

