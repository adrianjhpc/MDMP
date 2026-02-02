# MDMP: Managed Data Message Passing
This repository is intended for the development and documentation/description of the Managed Data Message Passing Approach.

MDMP is an alternative distributed memory programming approach that recognises some of the challenges of using an entirely prescriptive approach, such as MPI, for distributed applications.

Specifically, MDMP is designed to allow more variation in the granularity and timing of communications between collaborating processes, both to optimise the use of the network for a given program, and to reduce the overhead of implementing th message passing approach for programmers.

It is built on a compiler directives model, similar to the approach used by OpenMP, meaning a compiler generates the message passing functionality based on directives added to the program by the developer. In theory this should enable a seial version of the distributed memory program to be created and run without the directives being processed, as well as a distributed memory version with the directives.

MDMP is designed to work alongside existing MPI functionality, so it can be added incrementally to existing MPI programs as well as being used to develop distributed memory approaches from scratch.

MDMP is not designed to replace MPI entirely, indeed its functionality is built on the MPI library. It is more focussed on providing an alternative programming approach to using MPI that can enable more optimised, or at least more varied, communication patterns to be implemented without requiring significant code changes by developers and users.


