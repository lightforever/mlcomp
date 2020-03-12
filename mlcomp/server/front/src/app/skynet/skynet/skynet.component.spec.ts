import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { SkynetComponent } from './skynet.component';

describe('SkynetComponent', () => {
  let component: SkynetComponent;
  let fixture: ComponentFixture<SkynetComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ SkynetComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(SkynetComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
