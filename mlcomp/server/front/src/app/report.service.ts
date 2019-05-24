import {EventEmitter, Injectable} from '@angular/core';
import {BaseService} from "./base.service";

@Injectable({
  providedIn: 'root'
})
export class ReportService extends BaseService{
  protected collection_part: string = 'reports';
  protected single_part: string = 'report';

  data_updated: EventEmitter<any> = new EventEmitter();

}
