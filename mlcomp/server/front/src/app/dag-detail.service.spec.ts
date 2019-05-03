import { TestBed } from '@angular/core/testing';

import { DagDetailService } from './dag-detail.service';

describe('DagDetailService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: DagDetailService = TestBed.get(DagDetailService);
    expect(service).toBeTruthy();
  });
});
