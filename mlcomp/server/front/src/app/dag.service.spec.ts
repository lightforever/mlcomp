import { TestBed } from '@angular/core/testing';

import { DagService } from './dag.service';

describe('DagService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: DagService = TestBed.get(DagService);
    expect(service).toBeTruthy();
  });
});
