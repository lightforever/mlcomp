import { TestBed } from '@angular/core/testing';

import { ComputerService } from './computer.service';

describe('ComputerService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: ComputerService = TestBed.get(ComputerService);
    expect(service).toBeTruthy();
  });
});
